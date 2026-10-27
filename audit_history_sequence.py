"""Reproduce legacy left-padding/packed-length mismatch and verify corrected tokens."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
import torch
from history_sequence import pack_event_histories
from config import ARTIFACT_DIR as R

def main():
 torch.set_num_threads(1);source=Path('v21_experiments.py');tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pack_histories');namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
 lengths=np.array([0,1,2,4,7,10,15,20]);h=np.full((len(lengths),20),-1,dtype='int32')
 for row,n in enumerate(lengths):h[row,:n]=100+row*20+np.arange(n)
 rows=np.zeros((len(h),7),dtype='int32');rows[:,6]=np.arange(len(h));pad=5000;legacy,legacy_lengths=namespace['_pack_histories'](rows,h,pad);new,new_lengths,real=pack_event_histories(rows[:,6],h,pad)
 def packed_roundtrip(ids,n):
  sequence=torch.nn.utils.rnn.pack_padded_sequence(torch.from_numpy(ids),torch.from_numpy(n),batch_first=True,enforce_sorted=False);return torch.nn.utils.rnn.pad_packed_sequence(sequence,batch_first=True,padding_value=pad,total_length=20)[0].numpy()
 old_seen=packed_roundtrip(legacy,legacy_lengths);new_seen=packed_roundtrip(new,new_lengths);corrupted=[]
 for row,n in enumerate(lengths):
  assert np.array_equal(new_seen[row,:n],h[row,:n])
  if n and not np.array_equal(old_seen[row,:n],h[row,:n]):corrupted.append(int(n))
 assert corrupted==[1,2,4,7,10,15] and np.array_equal(real,lengths) and new_seen[0,0]==pad and new_lengths[0]==1
 eh=np.load(R/'evaluation_hist.npy',mmap_mode='r');el=(eh>=0).sum(1)
 report=dict(status='PASS',legacy_padding_bug_reproduced=True,legacy_corrupted_history_lengths=corrupted,corrected_all_test_histories_preserved=True,empty_history_requires_output_mask=True,evaluation_fraction_with_nonempty_short_history=float(np.mean((el>0)&(el<20))),legacy_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),corrected_source_sha256=hashlib.sha256(Path('history_sequence.py').read_bytes()).hexdigest(),scope='Packing correctness only, not a trained-model score. Historical GRU results remain recorded but do not validate a correctly packed sequence encoder.')
 (R/'history_sequence_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
