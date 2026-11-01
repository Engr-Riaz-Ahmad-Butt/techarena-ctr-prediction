"""nxb diagnostic: latent rank and history recency of the click-SVD cosine (day-6 cohort, ledger < cutoff day 6)."""
import numpy as np, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
R='artifacts/'; S=1737331200; DAY=86400; NI=50000; NU=54456
t=np.load(R+'train.npy'); hist=np.load(R+'train_hist.npy')
c=S+6*DAY; n=int(np.searchsorted(t[:,2],c)); P=t[:n]
pe=P[::10,6]; lh=hist[pe]; lu=np.repeat(P[::10,0].astype(np.int64),20); v=lh.ravel()>=0; m=P[:,5]>0
ui=np.unique(np.r_[lu[v],P[m,0]]*NI+np.r_[lh.ravel()[v],P[m,1]].astype(np.int64))
X=sp.csr_matrix((np.ones(len(ui)),(ui//NI,ui%NI)),shape=(NU,NI))
pop=np.asarray(X.sum(0)).ravel(); idf=np.log1p((X.getnnz(1)>0).sum()/np.maximum(pop,1))
Xw=X.multiply(idf[None,:]).tocsr(); rn=np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel()); rn[rn==0]=1; Xw=sp.diags(1/rn)@Xw
ref=np.load(R+'nxa_h1_day6.npz'); ev=ref['events'][::10]; y=ref['y']
H=hist[ev]; hv=H>=0; hl=hv.sum(1); cand=t[(ev[:,None]*10+np.arange(10)).ravel(),1].reshape(-1,10)
pos=np.arange(20)[None,:]
def prof(mask):
    Hx=sp.csr_matrix((idf[H[mask]],(np.repeat(np.arange(len(ev)),20)[mask.ravel()],H[mask])),shape=(len(ev),NI)); return Hx
masks={'all20':hv,'last10':hv&(pos>=hl[:,None]-10),'last5':hv&(pos>=hl[:,None]-5),'last2':hv&(pos>=hl[:,None]-2),'oldest10':hv&(pos<10)&(hl[:,None]>=20)|(hv&(hl[:,None]<20))}
for k in [4,8,12,16,24]:
    V=TruncatedSVD(k,n_iter=8,random_state=1).fit(Xw).components_.T
    Vc=V[cand]; nv=np.maximum(np.linalg.norm(Vc,axis=2),1e-9); out=[]
    for nm,mk in masks.items():
        Uh=prof(mk)@V; Uh/=np.maximum(np.linalg.norm(Uh,axis=1,keepdims=True),1e-9)
        s=np.einsum('ed,ecd->ec',Uh,Vc)/nv; out.append(f'{nm}:{roc_auc_score(y,s.ravel()):.4f}')
    print('k',k,' '.join(out),flush=True)
