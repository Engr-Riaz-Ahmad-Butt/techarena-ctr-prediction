"""Export V35 evaluation candidate using final-fit components and exact rank recipe.

Existing final V20/V21 components are reused, retaining their documented training
provenance. Fit the V23 outside outcome and V35 temporal residual before cutoff7.
The reported local AUC is validation evidence, not an expected official score.
"""
import csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R,EVALUATION_CSV
from v21_din import NI,SEED,START,DAY,train,hist,htimes,evaluation,ehist,etimes,pack_events,event_batches
from v23_outside import OutsideDIN,predict as predict_outside,rank
from v35_temporal_context import Context,pack as context_pack,batches as context_batches,predict as predict_context
N=529390

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def progress(stage,**kw):
 record=dict(stage=stage,**kw);(R/'v35_export_progress.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
def verify_local():
 audit=json.loads((R/'v35_temporal_context_audit.json').read_text());assert audit['accepted']
 for d in (6,5):
  z=np.load(R/f'v35_temporal_context_day{d}.npz');b=np.load(R/f'v23_outside_day{d}.npz')
  p=rank(rank(b['pred'])+.1*(rank(z['raw'])-rank(z['old'])))
  assert np.max(np.abs(p-z['pred']))<1e-12
  assert abs(roc_auc_score(z['y'],p)-audit['reports'][0 if d==6 else 1]['auc'])<1e-12
 return audit

def main():
 torch.set_num_threads(4);started=time.monotonic();audit=verify_local()
 output=Path('submission_v35.csv');assert not output.exists(),'Do not overwrite an existing submission.'
 inputs=[R/'submission_v20.csv',R/'v21_fastdin_final.pt',R/'v21_din_frozen_als_day6.pt',R/'imp2_model_warm1.0_d16_pc1_day5.npz']
 inputs += [R/f'v21_fastseeds_final_s{s}_eval.npy' for s in (0,1,2)]
 frozen=dict(input_sha256={p.name:sha(p) for p in inputs},source_sha256=sha(Path(__file__)),cutoff=START+7*DAY,threads=4,local_primary=audit['reports'][0]['auc'],local_secondary=audit['reports'][1]['auc'],note='Existing final V20/V21 artifacts reused. V21 final seeds use the earlier day6 frozen-ALS parent, as in their original finalizer; full parent retraining was not done here. Correct mean-of-ranks recipe is used without an extra rank of the seed mean. Local results are not an official-score forecast.')
 (R/'v35_export_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 assert sha(R/'submission_v20.csv')=='8f3bb5e49f96e49efc4d6416e9c593f27f118aef06c14e19b85ead45291c789b'
 cutoff=START+7*DAY;assert int(evaluation[:,2].min())>=cutoff
 rows=np.asarray(train[(train[:,2]>=START+4*DAY)&(train[:,2]<cutoff)])
 assert int(rows[:,2].max())<cutoff
 td=pack_events(rows,hist,htimes);ed=pack_events(np.asarray(evaluation),ehist,etimes);del rows
 torch.manual_seed(SEED);np.random.seed(SEED);model=OutsideDIN(NI)
 loaded=model.load_state_dict(torch.load(R/'v21_fastdin_final.pt',weights_only=True),strict=False)
 assert loaded.missing_keys==['outside'] and not loaded.unexpected_keys
 model.item.weight.requires_grad_(False);model.eval();totals=[]
 with torch.no_grad():
  for batch in event_batches(td,np.arange(min(td['n'],2048)),bs=512,shuffle=False):totals.append(torch.logsumexp(model.full_logits(batch)[:,:10],dim=1).numpy())
  rate=float(td['has_click'].mean());initial=float(np.median(np.concatenate(totals))+np.log((1-rate)/rate));model.outside.fill_(initial)
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=1e-5);model.train()
 progress('outside_training',events=td['n'],initial_outside=initial)
 for i,b in enumerate(event_batches(td,np.arange(td['n']),bs=512,shuffle=True,rng=np.random.default_rng(SEED+307)),1):
  target=torch.where(b['has_click'].bool(),b['target'],10);opt.zero_grad();loss=torch.nn.functional.cross_entropy(model.full_logits(b),target);loss.backward();opt.step()
  if i%100==0:progress('outside_training',batch=i,elapsed_s=time.monotonic()-started)
 torch.save(model.state_dict(),R/'v35_outside_final.pt');outside=predict_outside(model,ed);assert len(outside)==N and np.isfinite(outside).all()
 np.save(R/'v35_outside_eval.npy',outside)
 del td,ed,model,opt
 progress('context_training')
 with np.load(R/'imp2_model_warm1.0_d16_pc1_day5.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 rows=np.asarray(train[(train[:,2]>=START+5*DAY)&(train[:,2]<cutoff)])
 assert int(rows[:,2].min())>=START+5*DAY and int(rows[:,2].max())<cutoff
 td=context_pack(rows,hist,htimes,factors);del rows
 # Evaluation placeholders never enter a loss; only its histories and covariates are used.
 ed=context_pack(np.asarray(evaluation),ehist,etimes,factors)
 torch.manual_seed(3401);context=Context(factors['V'].shape[1]);opt=torch.optim.Adam(context.parameters(),lr=.003);rng=np.random.default_rng(3408)
 for epoch in range(1,4):
  context.train()
  for z,v,b,k,base,target in context_batches(td,factors,rng.permutation(td['n'])):
   opt.zero_grad();logits=context(z,v,b,k,base);loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(z),1)],1),target)
   (loss+.001*(context.bucket.square().sum()+context.kind.square().sum())).backward();opt.step()
  progress('context_training',epoch=epoch,elapsed_s=time.monotonic()-started)
 torch.save(context.state_dict(),R/'v35_context_final.pt');new,old=predict_context(context,ed,factors)
 np.save(R/'v35_context_eval.npy',new);np.save(R/'v35_context_old_eval.npy',old)
 with (R/'submission_v20.csv').open(newline='',encoding='utf-8') as f:base20=np.array([float(r['predicted_probability']) for r in csv.DictReader(f)])
 assert len(base20)==N
 seeds=[np.load(R/f'v21_fastseeds_final_s{s}_eval.npy') for s in (0,1,2)]
 assert all(len(a)==N and np.isfinite(a).all() for a in seeds)
 base21=.8*rank(base20)+.2*np.mean([rank(a) for a in seeds],axis=0)
 base23=.9*rank(base21)+.1*rank(outside)
 pred=rank(rank(base23)+.1*(rank(new)-rank(old)))
 assert len(pred)==N and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
 np.save(R/'v35_final_pred.npy',pred)
 progress('writing_csv')
 ids=set();id_hash=hashlib.sha256();temporary=output.with_suffix('.csv.tmp')
 with EVALUATION_CSV.open(newline='',encoding='utf-8-sig') as src,(R/'submission_v20.csv').open(newline='',encoding='utf-8') as prior,temporary.open('w',newline='',encoding='utf-8') as dest:
  writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability']);n=0
  for row,base,p in zip(csv.DictReader(src),csv.DictReader(prior),pred,strict=True):
   assert row['rowid']==base['rowid'] and row['rowid'] not in ids
   ids.add(row['rowid']);id_hash.update((row['rowid']+'\n').encode());writer.writerow([row['rowid'],format(float(p),'.15g')]);n+=1
 assert n==N;temporary.replace(output)
 for p in inputs:assert sha(p)==frozen['input_sha256'][p.name]
 manifest=dict(rows=n,output=str(output.resolve()),csv_sha256=sha(output),rowids_sha256=id_hash.hexdigest(),local_primary=frozen['local_primary'],local_secondary=frozen['local_secondary'],official_auc=None,cutoff=cutoff,training_last_timestamp=int(train[:,2].max()),evaluation_first_timestamp=int(evaluation[:,2].min()),artifact_sha256={p.name:sha(p) for p in [R/'v35_outside_final.pt',R/'v35_context_final.pt',R/'v35_outside_eval.npy',R/'v35_context_eval.npy',R/'v35_context_old_eval.npy',R/'v35_final_pred.npy']},note=frozen['note'])
 (R/'v35_submission_manifest.json').write_text(json.dumps(manifest,indent=2));progress('csv_created_pending_independent_check',rows=n,output=str(output.resolve()));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
