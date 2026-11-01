"""nxb diagnostic: is click behaviour non-stationary across days? (labels of days 0..6 only; no evaluation labels exist)"""
import numpy as np
from sklearn.metrics import roc_auc_score
t=np.load('artifacts/train.npy'); S=1737331200; DAY=86400; NI=50000
d=(t[:,2]-S)//DAY; it=t[:,1].astype(np.int64); y=t[:,5].astype(float)
imp=np.zeros((7,NI)); clk=np.zeros((7,NI))
for k in range(7):
    m=d==k; imp[k]=np.bincount(it[m],minlength=NI); clk[k]=np.bincount(it[m],y[m],minlength=NI)
g=y.mean()
# leave-one-out same-day item CTR vs previous-day and all-previous item CTR, scored on day 5 and 6 rows
for k in (5,6):
    m=d==k; i=it[m]; yy=y[m]
    loo=(clk[k][i]-yy+20*g)/(imp[k][i]-1+20)
    prev=(clk[k-1][i]+20*g)/(imp[k-1][i]+20)
    allp=(clk[:k].sum(0)[i]+20*g)/(imp[:k].sum(0)[i]+20)
    nxt=(clk[k+1][i]+20*g)/(imp[k+1][i]+20) if k<6 else None
    print(k,'same-day LOO item CTR auc',round(roc_auc_score(yy,loo),4),' prev-day',round(roc_auc_score(yy,prev),4),' all-previous',round(roc_auc_score(yy,allp),4), ' next-day', round(roc_auc_score(yy,nxt),4) if nxt is not None else '')
# per-item: distribution of shown counts across days (do items appear/disappear?)
shown=imp>0
print('items shown per day',shown.sum(1))
for k in range(1,7): print(k,'items shown on day k also shown on k-1:',round((shown[k]&shown[k-1]).sum()/shown[k].sum(),3),' rows on day k whose item was shown before day k:',round(np.isin(it[d==k],np.flatnonzero(imp[:k].sum(0)>0)).mean(),3))
