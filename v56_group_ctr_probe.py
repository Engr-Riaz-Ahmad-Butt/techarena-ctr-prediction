"""Probe the installed CPU backend's group-level CTR-history option on toy data."""
import json
from pathlib import Path
import numpy as np
from catboost import Pool,CatBoostClassifier,CatBoostError,__version__
def main():
 events=np.repeat(np.arange(50),10);x=np.array([[str(event//3)] for event in events],dtype=object);y=np.zeros(len(x),dtype='int32');y[np.arange(0,len(x),20)]=1;pool=Pool(x,label=y,cat_features=[0],group_id=events)
 try:
  model=CatBoostClassifier(iterations=2,depth=2,thread_count=4,ctr_history_unit='Group',random_seed=5601,allow_writing_files=False,verbose=False);model.fit(pool);result=dict(cpu_option_accepted=True)
 except CatBoostError as error:result=dict(cpu_option_accepted=False,error=str(error))
 result.update(catboost_version=__version__,scope='Capability probe only on toy events, not a competition model or proof of group-safe statistics semantics')
 path=Path(__file__).resolve().parent/'artifacts'/'v56_weekday'/'v56_group_ctr_probe.json';path.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
