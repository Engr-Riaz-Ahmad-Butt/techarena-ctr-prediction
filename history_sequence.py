"""Right-padded click sequences compatible with packed recurrent encoders."""
import numpy as np

def pack_event_histories(event_ids,histories,pad_id):
 """Return token IDs, pack lengths, and real lengths, preserving every click.

Real clicks must occupy the prefix, oldest to newest. An empty history gets one
dummy padding token for the packing API; encoders must mask its output using
real_lengths rather than treating that dummy as a click.
"""
 history=np.asarray(histories[event_ids]);assert history.ndim==2
 valid=history>=0;real_lengths=valid.sum(1).astype('int64');assert np.all(valid==(np.arange(history.shape[1])[None,:]<real_lengths[:,None]))
 assert np.all(history[valid]<pad_id)
 tokens=np.where(valid,history,pad_id).astype('int64');pack_lengths=np.maximum(real_lengths,1)
 return tokens,pack_lengths,real_lengths
