"""Refit one frozen auxiliary model and write a version-neutral candidate."""
from config import EVALUATION_CSV
from imp2_warm_choice import *

def finalize(dim):
    tag=f'warm1.0_d{dim}_pc1'
    selection=json.loads((ROOT/f'imp2_{tag}_selection.json').read_text())
    report=json.loads((ROOT/f'imp2_{tag}_report.json').read_text())
    assert selection['dimension']==dim and selection['pop_correct'] and selection['auxiliary_weight']==1.
    assert len(report)==2 and {r['day'] for r in report}=={5,6}
    for r in report:
        for key in ['dimension','pop_correct','auxiliary_weight','epoch','weight']:
            assert r[key]==selection[key],f'Unfrozen confirmation {key}'
        prior=np.load(ROOT/f"predictions_weighted_day{r['day']}.npz")
        reference=np.load(ROOT/f"predictions_v6_day{r['day']}.npz")
        eg=np.digitize((ehist>=0).sum(axis=1),[0,4,9,19],right=True)
        vg=np.digitize((hist[reference['events']]>=0).sum(axis=1),[0,4,9,19],right=True)
        ratios=(np.bincount(eg,minlength=5)/len(eg))/(np.bincount(vg,minlength=5)/len(vg))
        baseline_weighted=roc_auc_score(prior['y'],prior['pred'],sample_weight=ratios[vg])
        assert r['history_weighted_auc']>baseline_weighted,'Weighted confirmation must improve'
    assert int(train[:,2].max())<start+7*DAY
    assert all(r['auc']>r['baseline_auc'] for r in report),'Both dates must improve'
    users,items,history,target=event_data(train,hist)
    wu,wi=warm_data(train);counts=np.bincount(wi,minlength=ni)+1
    pop_bias=np.log(counts/counts.mean()).astype('float32')
    ps,acc=init32(start+6*DAY,dim);rng=np.random.default_rng(151)
    for ep in range(1,selection['epoch']+1):
        t=time.time();neg=rng.integers(0,ni,size=(len(wu),11),dtype=np.int32)
        collision=neg==wi[:,None]
        while collision.any():
            neg[collision]=rng.integers(0,ni,size=int(collision.sum()),dtype=np.int32)
            collision=neg==wi[:,None]
        warm_epoch(rng.permutation(len(wu)),wu,wi,neg,ps[0],ps[1],acc[0],acc[1],.045,.002,1.,pop_bias)
        loss=train_epoch(rng.permutation(len(users)),users,items,history,target,*ps,*acc,.045,.002)
        print('Final',tag,'epoch',ep,'loss',loss,'seconds',round(time.time()-t,1),flush=True)
    np.savez_compressed(ROOT/f'imp2_final_model_{tag}.npz',U=ps[0],V=ps[1],H=ps[2],bu=ps[3],bi=ps[4])
    raw=predict_events(evaluation[::10,0],evaluation[:,1].reshape(-1,10),ehist,*ps).ravel()
    assert len(raw)==len(evaluation) and np.isfinite(raw).all() and np.all((raw>=0)&(raw<=1))
    np.save(ROOT/f'imp2_final_raw_{tag}.npy',raw)
    source=EVALUATION_CSV
    dest=ROOT/f'imp2_submission_{tag}.csv';component=ROOT/f'imp2_component_{tag}.csv'
    weight=selection['weight'];seen=set()
    with (source.open(newline='',encoding='utf-8') as original,
          (ROOT/'submission_v11.csv').open(newline='') as old,
          dest.open('w',newline='') as output,component.open('w',newline='') as component_out):
        writer=csv.writer(output);cw=csv.writer(component_out)
        writer.writerow(['rowid','predicted_probability']);cw.writerow(['rowid','predicted_probability'])
        for row,previous,p in zip(csv.DictReader(original),csv.DictReader(old),raw,strict=True):
            assert row['rowid']==previous['rowid'] and row['rowid'] not in seen
            value=(1-weight)*float(previous['predicted_probability'])+weight*p
            assert np.isfinite(value) and 0<=value<=1
            seen.add(row['rowid']);writer.writerow([row['rowid'],format(value,'.10g')])
            cw.writerow([row['rowid'],format(p,'.10g')])
    assert len(seen)==529390
    import hashlib
    manifest=dict(tag=tag,selection=selection,confirmation=report[-1],rows=len(seen),
        baseline='submission_v11.csv',baseline_official_user_reported=.801,official_candidate_score=None,
        submission=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (ROOT/f'imp2_manifest_{tag}.json').write_text(json.dumps(manifest,indent=2))
    print('Candidate ready',dest,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dim',type=int,required=True);args=ap.parse_args();finalize(args.dim)
