"""nxb diagnostic: how does the history-profile cosine in a click-SVD space scale with the rank k? (day-6 cohort, cutoff day 6)"""
import numpy as np, scipy.sparse as sp, time
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
R='artifacts/'; S=1737331200; DAY=86400; NI=50000; NU=54456
t=np.load(R+'train.npy'); hist=np.load(R+'train_hist.npy'); htim=np.load(R+'train_times.npy')
c=S+6*DAY; n=int(np.searchsorted(t[:,2],c)); P=t[:n]
pe=P[::10,6]; lh=hist[pe]; lt=htim[pe]; lu=np.repeat(P[::10,0].astype(np.int64),20); v=lh.ravel()>=0
m=P[:,5]>0
U=np.r_[lu[v],P[m,0]]; I=np.r_[lh.ravel()[v],P[m,1]].astype(np.int64); T=np.r_[lt.ravel()[v],P[m,2]].astype(np.int64)
k=np.unique((U*NI+I)*(1<<31)+(T-1736000000)); ui=np.unique(k>>31); U=ui//NI; I=ui%NI
X=sp.csr_matrix((np.ones(len(U)),(U,I)),shape=(NU,NI))
pop=np.asarray(X.sum(0)).ravel(); idf=np.log1p((X.getnnz(1)>0).sum()/np.maximum(pop,1))
ref=np.load(R+'nxa_h1_day6.npz'); ev=ref['events'][::10]; y=ref['y']
H=hist[ev]; hv=H>=0; cand=t[(ev[:,None]*10+np.arange(10)).ravel(),1].reshape(-1,10)
Hx=sp.csr_matrix((idf[H[hv]],(np.repeat(np.arange(len(ev)),20)[hv.ravel()],H[hv])),shape=(len(ev),NI))
for wt in ['idf']:
  Xw=X.multiply(idf[None,:]).tocsr(); rn=np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel()); rn[rn==0]=1; Xw=sp.diags(1/rn)@Xw
  for kk in [16,32,64,128,256]:
    t0=time.time(); V=TruncatedSVD(kk,n_iter=6,random_state=1).fit(Xw).components_.T
    Uh=Hx@V; Uh/=np.maximum(np.linalg.norm(Uh,axis=1,keepdims=True),1e-9)
    Vc=V[cand]; s=np.einsum('ed,ecd->ec',Uh,Vc)/np.maximum(np.linalg.norm(Vc,axis=2),1e-9)
    d=np.einsum('ed,ecd->ec',Uh,Vc)
    print(wt,kk,'cos plain',round(roc_auc_score(y,s.ravel()),4),'dot plain',round(roc_auc_score(y,d.ravel()),4),f'{time.time()-t0:.0f}s',flush=True)
