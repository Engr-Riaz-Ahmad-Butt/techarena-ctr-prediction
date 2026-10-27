"""Read every ZIP member and verify the cloud upload package without extraction."""
import hashlib,json,zipfile
from pathlib import Path,PurePosixPath

def verify(path,notebook):
 path=Path(path);notebook=Path(notebook)
 with zipfile.ZipFile(path) as z:
  names=z.namelist();assert len(names)==len(set(names))
  for name in names:assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
  manifest=json.loads(z.read('bundle_manifest.json'));assert set(names)==set(manifest['files'])|{'bundle_manifest.json'}
  for i,(name,info) in enumerate(manifest['files'].items(),1):
   assert z.getinfo(name).file_size==info['bytes'];h=hashlib.sha256()
   with z.open(name) as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   assert h.hexdigest()==info['sha256'],name
   if i%10==0:print(f'Archive verified: {i}/{len(manifest["files"])} files',flush=True)
 nb=json.loads(notebook.read_text());assert nb['nbformat']==4
 for i,cell in enumerate(nb['cells']):
  assert cell['id']
  if cell['cell_type']=='code':compile(''.join(cell['source']),str(notebook)+':'+str(i),'exec')
 report=dict(status='PASS',archive_bytes=path.stat().st_size,verified_members=len(names),all_member_sha256_and_crc=True,notebook_code_compiles=True,scope='Local artifact/integration verification; Kaggle account and GPU execution not tested')
 (path.parent/'upload_verification.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]/'artifacts/kaggle_ready';print(json.dumps(verify(root/'techarena_kaggle_data.zip',root/'TechArena_CTR_Kaggle.ipynb')))
