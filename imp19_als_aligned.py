"""Causal aligned ALS coordinate interactions; existing summaries unchanged."""
from imp17_als_features import *


def unit(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def alignment(cut):
    assert cut>=2,'The anchor must precede the scored block'
    with np.load(ensure_model(2)) as m:
        assert int(m['cutoff'])==cutoff_of(2) and int(m['iteration'])==6
        anchor=unit(m['V'].astype(np.float64))
    with np.load(ensure_model(cut)) as m:
        assert int(m['cutoff'])==cutoff_of(cut) and int(m['iteration'])==6
        U,V,bu,bi,g,exposures=[m[k] for k in ['U','V','bu','bi','g','exposures']]
    left,singular,right=np.linalg.svd(unit(V).T@anchor,full_matrices=False)
    rotation=left@right
    assert np.max(np.abs(rotation.T@rotation-np.eye(16)))<1e-10
    return U,V,bu,bi,float(g),exposures,rotation,singular


def interactions(name,rows,histories,cut):
    assert cut>=2 and len(rows)%10==0
    assert np.all(rows[:,2]>=cutoff_of(cut)),'Future factor checkpoint for scored rows'
    destination=FEAT/f'imp19_als_aligned_{name}.npy'
    if destination.exists():
        x=np.load(destination,mmap_mode='r');assert x.shape==(len(rows),32);return x
    U,V,bu,bi,g,exposures,rotation,singular=alignment(cut)
    np.savez_compressed(ROOT/f'imp19_rotation_cut{cut}.npz',rotation=rotation,singular_values=singular,cutoff=cutoff_of(cut),anchor_cutoff=cutoff_of(2))
    times=np.load(ROOT/('evaluation_times.npy' if name=='eval' else 'train_times.npy'),mmap_mode='r')
    vr=unit(V@rotation);result=np.empty((len(rows),32),np.float32)
    for off in range(0,len(rows),20000):
        a=rows[off:off+20000];e=a[::10];items=a[:,1].reshape(-1,10)
        assert np.all(a[:,6].reshape(-1,10)==e[:,6,None])
        h=np.asarray(histories[e[:,6]]);valid=h>=0
        assert np.all((times[e[:,6]]<e[:,2,None])|~valid)
        u,_=score(a,U,V,bu,bi,g,exposures,return_profile=True,histories=histories,history_times=times)
        profile=(V[np.maximum(h,0)]*valid[:,:,None]).sum(1)/np.maximum(valid.sum(1)[:,None],1)
        ur=unit(u@rotation);hr=unit(profile@rotation);v=vr[items]
        result[off:off+len(a)]=np.concatenate([ur[:,None,:]*v,hr[:,None,:]*v],axis=2).reshape(-1,32)
    assert np.isfinite(result).all()
    np.save(destination,result)
    print('Aligned ALS interactions',name,result.shape,flush=True)
    return result
