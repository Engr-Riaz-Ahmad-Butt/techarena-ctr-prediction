"""Read-only Phase 1 diagnostics; no fitting, holdout scoring, or submissions."""
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,pearsonr
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as R

DAY=86400
def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
def correlation(x,y):
    return dict(pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic)) if np.std(x)>0 and np.std(y)>0 else dict(pearson=None,spearman=None)
def summary(x):
    return dict(n=len(x),mean=float(np.mean(x)),q10=float(np.quantile(x,.1)),median=float(np.median(x)),q90=float(np.quantile(x,.9))) if len(x) else dict(n=0)
def rank_distribution(values,labels):
    v=values.reshape(-1,10);yy=labels.reshape(-1,10);keep=yy.sum(1)==1;v=v[keep];p=yy[keep].argmax(1);clicked=v[np.arange(len(v)),p];above=(v>clicked[:,None]).sum(1);ties=(v==clicked[:,None]).sum(1)
    mass=np.array([np.mean(((r>above)&(r<=above+ties))/ties) for r in range(1,11)])
    return dict(clicked_events=len(v),rank1_is_highest=True,mean_rank=float((above+(ties+1)/2).mean()),rank_percent=(100*mass).tolist(),clicked_tie_share=float((ties>1).mean()))
def source(a,day,name,key):
    z=np.load(R/f'{name}_day{day}.npz');ev=z['events'];src=ev.astype(np.int64)*10+np.tile(np.arange(10),len(ev)//10);q=a['events'].astype(np.int64)*10+np.tile(np.arange(10),len(a['events'])//10);order=np.argsort(src);ix=order[np.searchsorted(src[order],q)];assert np.array_equal(src[ix],q);return z[key][ix]
def activity_gap(t,rows,cut,nu):
    prior=t[t[:,2]<cut];last=np.zeros(nu,dtype=np.int32);np.maximum.at(last,prior[:,0],prior[:,2]);users=rows[::10,0];valid=last[users]>0;gap=(cut-last[users[valid]])/DAY
    return dict(users=len(users),no_prior_training_activity=int((~valid).sum()),gap_days=summary(gap),within_1_day=float(np.mean(valid&(cut-last[users]<=DAY))),within_2_days=float(np.mean(valid&(cut-last[users]<=2*DAY))),within_3_days=float(np.mean(valid&(cut-last[users]<=3*DAY))))
def id_groups(ids,rows,y):
    out={}
    for name,col in [('users',0),('items',1)]:
        raw=np.zeros(len(ids[name]),dtype=float)
        for k,v in ids[name].items():raw[v]=int(k.split('_',1)[1][:12],16)/16**12
        vals=raw[rows[:,col]];bins=np.minimum((vals*10).astype(int),9);groups=[]
        for b in range(10):
            m=bins==b;groups.append(dict(hash_decile=b,rows=int(m.sum()),click_rate=float(y[m].mean()) if m.any() else None,mean_time=float(rows[m,2].mean()) if m.any() else None))
        out[name]=dict(raw_hash_vs_label=correlation(vals,y),raw_hash_vs_timestamp=correlation(vals,rows[:,2]),encoded_index_vs_label=correlation(rows[:,col],y),encoded_index_vs_timestamp=correlation(rows[:,col],rows[:,2]),hash_deciles=groups)
    return out

def main():
    tic=time.monotonic();t=np.load(R/'train.npy',mmap_mode='r');h=np.load(R/'train_hist.npy',mmap_mode='r');e=np.load(R/'evaluation.npy',mmap_mode='r');ids=json.loads((R/'ids.json').read_text());ni=len(ids['items']);nu=len(ids['users']);start=int(t[:,2].min());trainend=int(t[:,2].max())+60;assert trainend==start+7*DAY
    first=np.full(ni,np.iinfo(np.int32).max,dtype=np.int32);np.minimum.at(first,t[:,1],t[:,2]);reports=[];gaps=[]
    fixed_prior=np.asarray(t[t[:,2]<start+5*DAY]);fixed_pops={str(w):np.bincount(fixed_prior[fixed_prior[:,2]>=start+5*DAY-w*DAY,1],minlength=ni) for w in (1,3,7)};fixed_pops['all']=np.bincount(fixed_prior[:,1],minlength=ni)
    for day in (5,6):
        z=np.load(R/f'selection_day{day}.npz');events=z['events'];rows=np.asarray(t[events*10+np.tile(np.arange(10),len(events)//10)]);assert np.array_equal(rows[:,6],events);y=z['y'];base=z['baseline'];cut=start+day*DAY;prior=np.asarray(t[t[:,2]<cut]);items=rows[:,1];n=np.bincount(prior[:,1],minlength=ni);clicks=np.bincount(prior[:,1],weights=prior[:,5],minlength=ni);scores={};popauc={};recent={}
        for w in (1,3,7):
            p=np.bincount(prior[prior[:,2]>=cut-w*DAY,1],minlength=ni);recent[w]=p;scores[f'pop_last{w}']=p[items].astype(float);popauc[f'last{w}']=auc(y,p[items])
        scores['pop_all']=n[items].astype(float);popauc['all']=auc(y,n[items]);scores['click_pop_all']=clicks[items]
        seen=first[items]<cut;age=np.where(seen,(rows[:,2]-first[items])/DAY,0);scores['item_age_older_higher']=age;scores['item_age_newer_higher']=-age
        scores['ALS']=source(z,day,'imp15_als','raw')
        # Exposure-normalized recent share / all-time share; smoothing protects zero support.
        ratio=((recent[3][items]+1)/(recent[3].sum()+ni))/((n[items]+1)/(n.sum()+ni));share=n[items]/len(prior);scores['recent3_all_share_ratio']=ratio
        lifecycle={}
        for name,values in [('age_days',age),('recent3_to_all_popularity_share_ratio',ratio),('alltime_impression_share',share),('last3_impression_share',recent[3][items]/max(recent[3].sum(),1))]:lifecycle[name]=dict(clicked=summary(values[y==1]),not_clicked=summary(values[y==0]),auc=auc(y,values))
        cold=~seen;young=seen&(first[items]>=cut-2*DAY)
        coldreport={name:dict(rows=int(mask.sum()),share=float(mask.mean()),clicks=int(y[mask].sum()),v53_auc=auc(y[mask],base[mask])) for name,mask in [('unseen_before_cutoff',cold),('first_seen_last2days',young),('unseen_or_recent',cold|young)]}
        # V53 stores ranks, not calibrated probabilities. Both residual definitions reported.
        rankres=y-base;centeredres=y-float(y.mean());simple=[]
        for name,value in scores.items():simple.append(dict(signal=name,auc=auc(y,value),vs_label_minus_V53_rank=correlation(value,rankres),vs_label_minus_global_ctr=correlation(value,centeredres)))
        ranked={name:rank_distribution(scores[key],y) for name,key in [('alltime_popularity','pop_all'),('last3_popularity','pop_last3'),('older_item_age','item_age_older_higher'),('ALS','ALS')]}
        clicked=y.reshape(-1,10).sum(1)==1;clickedpos=y.reshape(-1,10)[clicked].argmax(1);position=dict(clicked_events=int(clicked.sum()),position_counts=np.bincount(clickedpos,minlength=10).tolist(),position_percent=(100*np.bincount(clickedpos,minlength=10)/len(clickedpos)).tolist(),per_position_click_rate=y.reshape(-1,10).mean(0).tolist())
        # Hourly slices are descriptive, not multi-day drift from one frozen V53 model.
        slices=[]
        for b in range(4):
            mask=(rows[:,2]-cut>=b*DAY/4)&(rows[:,2]-cut<(b+1)*DAY/4);slices.append(dict(hours_after_cutoff=[b*6,(b+1)*6],rows=int(mask.sum()),v53_auc=auc(y[mask],base[mask])))
        fixed=[dict(day_after_fixed_day5_popularity_cutoff=day-4,window=k,auc=auc(y,p[items])) for k,p in fixed_pops.items()]
        # Exposure distribution of negatives vs the prior catalog exposure distribution.
        boundaries=np.quantile(np.log1p(n),np.linspace(0,1,11));bins=np.searchsorted(boundaries[1:-1],np.log1p(n),side='right');candidate_b=np.bincount(bins[items],minlength=10)/len(items);uniform_b=np.bincount(bins,minlength=10)/ni;weighted_b=np.bincount(bins,weights=n,minlength=10)/n.sum();negative_b=np.bincount(bins[items[y==0]],minlength=10)/int((y==0).sum())
        negsampling=dict(log_pop_decile_boundaries=boundaries.tolist(),candidate_share=candidate_b.tolist(),negative_share=negative_b.tolist(),uniform_catalog_share=uniform_b.tolist(),impression_weighted_catalog_share=weighted_b.tolist(),total_variation_negative_vs_uniform=float(.5*abs(negative_b-uniform_b).sum()),total_variation_negative_vs_impression_weighted=float(.5*abs(negative_b-weighted_b).sum()),note='Prior impression popularity already reflects the unknown generator; descriptive fit is not proof of sampling mechanism.')
        gaps.append(dict(day=day,**activity_gap(t,rows,cut,nu)))
        zero=(h[events]>=0).sum(1)==0
        report=dict(day=day,rows=len(rows),events=len(rows)//10,cutoff=cut,v53_auc=auc(y,base),popularity_auc=popauc,lifecycle=lifecycle,cold_items=coldreport,simple_scores=simple,clicked_rank_distributions=ranked,clicked_list_position=position,cutoff_hour_slices=slices,fixed_cutoff_popularity_drift=fixed,negative_sampling=negsampling,id_patterns=id_groups(ids,rows,y),zero_history_age_auc=auc(y[zero],age[zero]),zero_history_rows=int(zero.sum()))
        reports.append(report);print(json.dumps(dict(stage='validation_diagnostics',day=day,auc=report['v53_auc'],popularity_auc=popauc,cold=coldreport)),flush=True)
    # Evaluation uses only IDs and timestamps. No score, label, or fit is accessed.
    ei=e[:,1];seen=first[ei]<trainend;young=seen&(first[ei]>=trainend-2*DAY);evalcold=dict(rows=len(e),unseen=int((~seen).sum()),first_seen_last2days=int(young.sum()),unseen_or_recent=int(((~seen)|young).sum()))
    evalgap=activity_gap(t,np.asarray(e),trainend,nu)
    # Across-user event structure is descriptive and uses no training labels.
    head=np.asarray(t[::10]);order=np.lexsort((head[:,2],head[:,0]));last={};slates=set();adj_intersections=[];exact=0;presentations=0;repeated=0;useritems={}
    for ev in order:
        u=int(head[ev,0]);items=set(map(int,t[ev*10:ev*10+10,1]));previous=last.get(u)
        if previous is not None:
            overlap=len(items&previous);adj_intersections.append(overlap);exact+=int(items==previous)
        known=useritems.setdefault(u,set());repeated+=len(items&known);presentations+=10;known.update(items);last[u]=items
    structure=dict(events=len(head),users=len(useritems),consecutive_user_event_pairs=len(adj_intersections),consecutive_overlap_count=summary(np.asarray(adj_intersections)),consecutive_any_overlap_share=float(np.mean(np.asarray(adj_intersections)>0)),consecutive_identical_slate_count=exact,user_item_repeat_presentation_share=repeated/presentations,note='All training candidate IDs/timestamps only; labels not inspected for this structure calculation.')
    out=dict(scope='Full saved selection days: day5 352300 rows; day6 244220 rows. Reserved holdout never opened/scored.',validation=reports,evaluation_cold_counts=evalcold,user_activity_gap=dict(validation=gaps,evaluation=evalgap),user_event_structure=structure,drift_limitation='No frozen V53 model has saved labeled predictions for successive future days. Day5 is user-OOF and day6 uses a later head, so those AUCs are not day1/day2 drift of the same model. Hourly slices and fixed day5 popularity across days1/2 are available. Days3+ after cutoff require additional labeled dates and saved predictions.',residual_limitation='V53 predictions are ranks, not calibrated probabilities; label minus V53 score is a rank residual, so correlation can reflect score scaling. Signals selected from these reused dates are diagnostic hypotheses only.',item_age_definition='First training-impression timestamp, not catalog creation time. Diagnostic scores use pre-cutoff history only; first appearance on/after cutoff is unseen and age set to zero.',elapsed_seconds=time.monotonic()-tic)
    (R/'v54_phase1_diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(stage='complete',elapsed_seconds=out['elapsed_seconds'],evaluation_cold_counts=evalcold,user_event_structure=structure,user_activity_gap=out['user_activity_gap'])),flush=True)

if __name__=='__main__':main()
