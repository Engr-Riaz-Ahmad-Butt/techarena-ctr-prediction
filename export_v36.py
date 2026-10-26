"""Final-fit the V36 recent-interest correction; reuse verified V35 backbone."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from config import ARTIFACT_DIR as R
from v21_din import START,DAY,train,hist,htimes,evaluation,ehist,etimes
from v36_recent_context import Context,pack,batches,predict,rank

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);torch.manual_seed(3401)
 audit=json.loads((R/'v36_recent_context_audit.json').read_text());assert audit['accepted']
 prior=json.loads((R/'v35_submission_checks.json').read_text());assert prior['status']=='PASS' and sha(Path('submission_v35.csv'))==prior['csv_sha256']
 output=Path('submission_v36.csv');assert not output.exists()
 inputs=[R/'submission_v20.csv',R/'v35_outside_eval.npy',R/'v35_outside_final.pt',R/'imp2_model_warm1.0_d16_pc1_day5.npz']+[R/f'v21_fastseeds_final_s{s}_eval.npy' for s in (0,1,2)]
 frozen=dict(input_sha256={p.name:sha(p) for p in inputs},source_sha256=sha(Path(__file__)),baseline_csv_sha256=sha(Path('submission_v35.csv')),cutoff=START+7*DAY,local_primary=audit['reports'][0]['auc'],local_secondary=audit['reports'][1]['auc'],note='V36 recent-interest component refit before evaluation cutoff. Verified existing V35/V21 final backbone reused with its previously documented parent-checkpoint provenance; local AUC is not an official-score forecast.')
 (R/'v36_export_frozen.json').open('x').write(json.dumps(frozen,indent=2))
 with np.load(R/'imp2_model_warm1.0_d16_pc1_day5.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 rows=np.asarray(train[(train[:,2]>=START+5*DAY)&(train[:,2]<START+7*DAY)]);assert rows[:,2].max()<START+7*DAY
 td=pack(rows,hist,htimes,factors);ed=pack(np.asarray(evaluation),ehist,etimes,factors)
 model=Context(factors['V'].shape[1]);optimizer=torch.optim.Adam(model.parameters(),lr=.003);rng=np.random.default_rng(3408)
 for epoch in range(1,4):
  model.train()
  for z,v,b,k,base,target in batches(td,factors,rng.permutation(td['n'])):
   optimizer.zero_grad();logits=model(z,v,b,k,base)
   loss=torch.nn.functional.cross_entropy(torch.cat([logits,torch.zeros(len(z),1)],1),target)
   penalty=.001*(model.bucket.square().sum()+model.kind.square().sum()+model.recent.square().sum())
   (loss+penalty).backward();optimizer.step()
  print(json.dumps(dict(stage='v36_final_training',epoch=epoch)),flush=True)
 torch.save(model.state_dict(),R/'v36_context_final.pt');new,old=predict(model,ed,factors)
 np.save(R/'v36_context_eval.npy',new);np.save(R/'v36_context_old_eval.npy',old)
 with (R/'submission_v20.csv').open(newline='',encoding='utf-8') as f:b=np.array([float(row['predicted_probability']) for row in csv.DictReader(f)])
 seeds=[np.load(R/f'v21_fastseeds_final_s{s}_eval.npy') for s in (0,1,2)]
 base21=.8*rank(b)+.2*np.mean([rank(a) for a in seeds],axis=0)
 base23=.9*rank(base21)+.1*rank(np.load(R/'v35_outside_eval.npy'))
 pred=rank(rank(base23)+.1*(rank(new)-rank(old)));assert len(pred)==529390 and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
 np.save(R/'v36_final_pred.npy',pred)
 temporary=output.with_suffix('.csv.tmp');ids=set();ih=hashlib.sha256()
 with Path('submission_v35.csv').open(newline='',encoding='utf-8') as source,temporary.open('w',newline='',encoding='utf-8') as dest:
  w=csv.writer(dest);w.writerow(['rowid','predicted_probability']);n=0
  for row,p in zip(csv.DictReader(source),pred,strict=True):
   key=row['rowid'];assert key not in ids;ids.add(key);ih.update((key+'\n').encode());w.writerow([key,format(float(p),'.15g')]);n+=1
 assert n==529390;temporary.replace(output)
 for p in inputs:assert sha(p)==frozen['input_sha256'][p.name]
 artifacts=[R/'v35_outside_final.pt',R/'v35_outside_eval.npy',R/'v36_context_final.pt',R/'v36_context_eval.npy',R/'v36_context_old_eval.npy',R/'v36_final_pred.npy']
 manifest=dict(output=str(output.resolve()),rows=n,csv_sha256=sha(output),rowids_sha256=ih.hexdigest(),local_primary=frozen['local_primary'],local_secondary=frozen['local_secondary'],official_auc=None,artifact_sha256={p.name:sha(p) for p in artifacts},training_last_timestamp=int(rows[:,2].max()),cutoff=START+7*DAY,evaluation_first_timestamp=int(evaluation[:,2].min()),note=frozen['note'])
 (R/'v36_submission_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
