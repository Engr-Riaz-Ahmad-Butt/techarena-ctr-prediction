from config import TRAIN_CSV, EVALUATION_CSV
import csv,json,time
from core import ROOT,np,train,hist,evaluation,ehist

for name,path,data,h in [
    ('train',TRAIN_CSV,train,hist),
    ('evaluation',EVALUATION_CSV,evaluation,ehist)]:
    dest=np.lib.format.open_memmap(ROOT/(name+'_times.npy'),mode='w+',dtype='int32',shape=h.shape)
    dest[:]=0
    begin=time.time()
    with path.open(newline='',encoding='utf-8') as f:
        for n,row in enumerate(csv.DictReader(f)):
            if n%10==0:
                e=n//10; times=json.loads(row['history_timestamps'])
                assert len(times)==int((h[e]>=0).sum())
                assert all(t<int(row['timestamp']) for t in times)
                assert times==sorted(times)
                assert int(row['timestamp'])==data[n,2]
                dest[e,:len(times)]=times
            if (n+1)%2000000==0:print(name,n+1,'rows',round(time.time()-begin),'seconds',flush=True)
    assert n+1==len(data)
    dest.flush();print(name,'timestamps verified',len(dest),'events',flush=True)
