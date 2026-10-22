"""A fixed small rank-ensemble search using previously fitted, causal members."""
from imp3_lib import *
from scipy.stats import rankdata
from tst2_candidate import influence,weighted_influence


def members(day,reference):
    specs=dict(leafwise=(f'imp10_leafwise_raw_day{day}.npz','raw'),
               warmcontext=(f'predictions_imp5_warmcontext_day{day}.npz','raw'),
               bilinear=(f'imp11_bilinear_day{day}.npz','raw'),
               v6=(f'predictions_v6_day{day}.npz','pred'),
               v7=(f'predictions_v7_day{day}.npz','pred'))
    result={}
    for name,(filename,column) in specs.items():
        data=np.load(ROOT/filename)
        assert np.array_equal(data['events'],reference['events']) and np.array_equal(data['y'],reference['y'])
        result[name]=data[column]
    return result


def run(day,chosen=None):
    baseline=np.load(ROOT/f'imp15_als_day{day}.npz')
    y=baseline['y'];events=baseline['events'];base=baseline['pred']
    sw,_=history_weights(events)
    old=float(roc_auc_score(y,base));oldw=float(roc_auc_score(y,base,sample_weight=sw))
    ranked=rankdata(base,method='average')/(len(base)+1)
    sources=members(day,baseline);best=None;trials=[]
    for name,raw in sources.items():
        if chosen and name!=chosen['member']:continue
        r=rankdata(raw,method='average')/(len(raw)+1)
        for weight in ([chosen['weight']] if chosen else [.05,.1,.2,.3]):
            pred=(1-weight)*ranked+weight*r
            auc=float(roc_auc_score(y,pred));weighted=float(roc_auc_score(y,pred,sample_weight=sw))
            result=dict(day=day,member=name,weight=weight,auc=auc,reweighted=weighted,baseline_auc=old,baseline_reweighted=oldw)
            trials.append(result);print(json.dumps(result),flush=True)
            if chosen or (auc>old and weighted>oldw and (best is None or auc>best['auc'])):
                best=result.copy();se=np.linalg.norm(influence(y,pred)-influence(y,base))
                sew=np.linalg.norm(weighted_influence(y,pred,sw)-weighted_influence(y,base,sw))
                best.update(delta=auc-old,delta_reweighted=weighted-oldw,paired_interval=[auc-old-1.96*se,auc-old+1.96*se],weighted_paired_interval=[weighted-oldw-1.96*sew,weighted-oldw+1.96*sew])
                np.savez_compressed(ROOT/f'imp16_rankblend_day{day}.npz',y=y,events=events,raw=raw,pred=pred)
    (ROOT/f'imp16_rankblend_trials_day{day}.json').write_text(json.dumps(trials,indent=2))
    return best


if __name__=='__main__':
    output=ROOT/'imp16_rankblend_report.json';assert not output.exists(),'Do not restart a completed experiment.'
    selected=run(6);reports=[selected];output.write_text(json.dumps(reports,indent=2))
    if selected and selected['paired_interval'][0]>0 and selected['weighted_paired_interval'][0]>0:
        reports.append(run(5,selected));output.write_text(json.dumps(reports,indent=2))
    print('Rank ensemble complete; independent both-date verification required.',flush=True)
