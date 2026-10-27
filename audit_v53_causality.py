"""Counterfactual-label checks on sampled full-event propensity features."""
import hashlib,json
import numpy as np
from config import ARTIFACT_DIR as R
import v53_causal_propensity as candidate

def main():
 import pickle
 t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');ht=np.load(R/'train_times.npy',mmap_mode='r');day=5;cut=int(t[:,2].min())+day*86400
 with (R/'v48_history_calibrator.pkl').open('rb') as f:spline=pickle.load(f)['spline']
 original_function=candidate.prior_features;z,x,users,groups,names=original_function(day,t,h,ht,spline);sample=np.arange(0,len(users)//10,max(1,len(users)//10//64));ix=(sample[:,None]*10+np.arange(10)).ravel();fixture_z={k:z[k][ix].copy() for k in ['y','events','pred']};fixture=(fixture_z,x[ix].copy(),users[ix].copy(),groups[ix].copy(),names);width=x.shape[1];del x
 candidate.prior_features=lambda *args:fixture
 _,actual,_,_,feature_names=candidate.features(day,t,h,ht,spline)
 changed=np.asarray(t).copy();future=t[:,2]>=cut;labels=changed[future,5].reshape(-1,10);changed[future,5]=np.roll(labels,1,axis=1).ravel();changes=int(np.count_nonzero(changed[future,5]!=t[future,5]));assert changes>0
 _,negative,_,_,_=candidate.features(day,changed,h,ht,spline);assert np.array_equal(actual,negative)
 # Oracle checks one feature directly from raw event records.
 u=int(fixture[2][0]);prior_events=t[::10];expected=float(np.log1p(((prior_events[:,0]==u)&(prior_events[:,2]<cut)).sum()));column=width+feature_names[len(names):].index('log_user_prior_events');assert abs(actual[0,column]-expected)<1e-12
 changed[:,5]=t[:,5];eligible=np.flatnonzero(np.isin(t[::10,0],fixture[2])&(t[::10,2]<cut));assert len(eligible)>0;event=int(eligible[0]);ys=changed[event*10:event*10+10,5];had=int(ys.sum());ys[:]=0
 if had==0:ys[0]=1
 _,positive,_,_,_=candidate.features(day,changed,h,ht,spline);effect=float(np.max(np.abs(positive[:,width:]-actual[:,width:])));assert effect>0
 candidate.prior_features=original_function
 result=dict(status='PASS',sampled_events=len(sample),heldout_labels_changed=changes,heldout_label_change_max_feature_effect=float(np.max(np.abs(negative-actual))),prior_label_change_max_feature_effect=effect,user_event_count_oracle=True,scope='Counterfactual arrays only; source data files were not modified. Shows new features exclude held-out labels and respond to prior labels.')
 (R/'v53_causality_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
