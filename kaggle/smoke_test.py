"""Local synthetic CPU integration test; never a competition score."""
import json,tempfile
from pathlib import Path
import numpy as np
import runner

def main():
 parent=Path(__file__).resolve().parents[1]/'artifacts'
 with tempfile.TemporaryDirectory(dir=parent,prefix='kaggle-smoke-') as directory:
  root=Path(directory).resolve();assert root.is_relative_to(parent.resolve());rng=np.random.default_rng(7001);manifest=dict(schema=1,feature_width=354,train={},validation={},files={});widths=[190,4,32,32,32,32,32]
  def record(path):manifest['files'][path.name]=dict(bytes=path.stat().st_size,sha256=runner.digest(path));return path.name
  def features(name,n):
   names=[]
   for k,width in enumerate(widths):
    path=root/f'{name}_{k}.npy';np.save(path,rng.normal(size=(n,width)).astype('float32'));names.append(record(path))
   return names
  for day in [2,3,4,5]:
   y=np.zeros(200,dtype='float32');y[::10]=1;path=root/f'y{day}.npy';np.save(path,y);meta=root/f'meta{day}.json';meta.write_text(json.dumps(dict(last_timestamp=day)))
   manifest['train'][str(day)]=dict(labels=record(path),metadata=record(meta),features=features('train'+str(day),len(y)))
  for day in [5,6]:
   y=np.zeros(100,dtype='float32');y[::10]=1;path=root/f'val{day}.npz';np.savez(path,y=y,events=np.repeat(np.arange(10),10),baseline=.1+.8*y,weights=np.ones(len(y)))
   manifest['validation'][str(day)]=dict(cutoff=day,data=record(path),features=features('val'+str(day),len(y)))
  (root/'bundle_manifest.json').write_text(json.dumps(manifest));runner.load_bundle(root)
  result=runner.run(root,root/'result',device='cpu',rounds=3);assert result['reports'][0]['history_adjusted_auc']==1 and not result['eligible_for_review'] and not result['csv_ready'];assert not (root/'result/day5').exists()
  before=runner.digest(root/'result/day6/model.ubj');runner.run(root,root/'result',device='cpu',rounds=3,verify=False);assert runner.digest(root/'result/day6/model.ubj')==before
  runner.experiment(root,manifest,root/'confirmation',day=5,device='cpu',rounds=3)
  archive=runner.archive_results(root/'result',root/'download.zip');assert archive.is_file()
  path=root/manifest['train']['2']['features'][0]
  with path.open('ab') as f:f.write(b'corruption')
  try:runner.load_bundle(root)
  except ValueError:pass
  else:raise AssertionError('Corrupted bundle was accepted')
 report=dict(status='PASS',scope='Synthetic local CPU integration only; not Kaggle/GPU execution or competition score',checks=['both main validation days','file checksum rejection','data iterator across multiple training days','completed model replay without refitting','report and result archive creation','no CSV output or automatic promotion'])
 target=Path(__file__).resolve().parents[1]/'results/kaggle_smoke_test.json';target.write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
