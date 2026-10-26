"""Independent schema/row-ID/recipe checks and final-model sample replay for V35."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from config import ARTIFACT_DIR as R,EVALUATION_CSV
from v23_outside import OutsideDIN,predict as outside_predict,rank
from v21_din import NI,evaluation,ehist,etimes,pack_events
from v35_temporal_context import Context,pack,predict

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);out=Path('submission_v35.csv');m=json.loads((R/'v35_submission_manifest.json').read_text());frozen=json.loads((R/'v35_export_frozen.json').read_text());assert sha(out)==m['csv_sha256']
 for n,h in frozen['input_sha256'].items():assert sha(R/n)==h
 for n,h in m['artifact_sha256'].items():assert sha(R/n)==h
 with (R/'submission_v20.csv').open(newline='',encoding='utf-8') as f:b=np.array([float(r['predicted_probability']) for r in csv.DictReader(f)])
 seeds=[np.load(R/f'v21_fastseeds_final_s{s}_eval.npy') for s in (0,1,2)]
 outside=np.load(R/'v35_outside_eval.npy');new=np.load(R/'v35_context_eval.npy');old=np.load(R/'v35_context_old_eval.npy')
 for a in [outside,new,old]:assert np.isfinite(a).all() and (a>=0).all() and (a.reshape(-1,10).sum(1)<=1.000001).all()
 base21=.8*rank(b)+.2*np.mean([rank(a) for a in seeds],axis=0);base23=.9*rank(base21)+.1*rank(outside)
 expected=rank(rank(base23)+.1*(rank(new)-rank(old)))
 assert np.array_equal(expected,np.load(R/'v35_final_pred.npy'))
 ids=set();ih=hashlib.sha256();error=0.;n=0
 with out.open(newline='',encoding='utf-8') as dest,EVALUATION_CSV.open(newline='',encoding='utf-8-sig') as src:
  reader=csv.DictReader(dest);assert reader.fieldnames==['rowid','predicted_probability']
  for row,original,p in zip(reader,csv.DictReader(src),expected,strict=True):
   assert row['rowid']==original['rowid'] and row['rowid'] not in ids
   ids.add(row['rowid']);ih.update((row['rowid']+'\n').encode());value=float(row['predicted_probability'])
   assert np.isfinite(value) and 0<=value<=1;error=max(error,abs(value-p));n+=1
 assert n==529390 and ih.hexdigest()==m['rowids_sha256'] and error<1e-14
 sample=np.asarray(evaluation[:1000]);model=OutsideDIN(NI);model.load_state_dict(torch.load(R/'v35_outside_final.pt',weights_only=True));replay=outside_predict(model,pack_events(sample,ehist,etimes));outside_error=float(np.max(np.abs(replay-outside[:1000])));assert outside_error<1e-6
 with np.load(R/'imp2_model_warm1.0_d16_pc1_day5.npz') as f:factors={k:f[k].astype('float32') for k in ['U','V','H','bu','bi']}
 model=Context(factors['V'].shape[1]);model.load_state_dict(torch.load(R/'v35_context_final.pt',weights_only=True));nr,orr=predict(model,pack(sample,ehist,etimes,factors),factors)
 context_error=max(float(np.max(np.abs(nr-new[:1000]))),float(np.max(np.abs(orr-old[:1000]))));assert context_error<1e-6
 assert m['training_last_timestamp']<m['cutoff']<=m['evaluation_first_timestamp']
 result=dict(status='PASS',rows=n,unique_rowids=len(ids),csv_sha256=sha(out),maximum_csv_rounding_error=error,outside_replay_error=outside_error,context_replay_error=context_error,official_auc=None,local_reference_auc=m['local_primary'],note=m['note'])
 (R/'v35_submission_checks.json').write_text(json.dumps(result,indent=2));(R/'v35_export_progress.json').write_text(json.dumps(dict(stage='complete_verified_csv',output=str(out.resolve()),rows=n),indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
