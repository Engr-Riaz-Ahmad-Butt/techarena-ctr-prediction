"""Pre-cutoff warm-sequence and contextual event choice factor objectives."""
from imp2_warm_choice import *


@njit
def sequence_examples(users,history,stamps):
    maximum=int((history>=0).sum());n=0
    us=np.empty(maximum,np.int32);items=np.empty(maximum,np.int32)
    contexts=np.full((maximum,20),-1,np.int32)
    buckets=np.empty(maximum,np.int32);types=np.empty(maximum,np.int32)
    for row in range(len(users)):
        for q in range(20):
            item=history[row,q];stamp=stamps[row,q]
            if item<0:continue
            duplicate=False
            for j in range(q):
                if history[row,j]==item and stamps[row,j]==stamp:duplicate=True
            if duplicate:continue
            us[n]=users[row];items[n]=item;buckets[n]=stamp%86400//21600
            types[n]=int(((stamp//86400+3)%7)>=5)
            count=0
            for j in range(q):
                if history[row,j]<0 or stamps[row,j]>=stamp:continue
                duplicate_prior=False
                for k in range(j):
                    if history[row,k]==history[row,j] and stamps[row,k]==stamps[row,j]:duplicate_prior=True
                if not duplicate_prior:
                    contexts[n,count]=history[row,j];count+=1
            n+=1
    return us[:n],items[:n],contexts[:n],buckets[:n],types[:n]


@njit
def profile(u,history,U,H):
    n=0
    for h in history:
        if h>=0:n+=1
    scale=1/math.sqrt(max(n,1));z=U[u].copy()
    for h in history:
        if h>=0:z+=scale*H[h]
    return z,scale


@njit
def apply_step(u,items,history,bucket,kind,z,scale,errors,
               U,V,H,bu,bi,B,T,au,av,ah,abu,abi,aB,aT,lr,reg,ctxreg,event_bias):
    d=U.shape[1];gradient=np.zeros(d,np.float64)
    # Accumulate ALL z gradients before changing any shared item vector.
    for j in range(len(items)):
        for k in range(d):gradient[k]+=errors[j]*V[items[j],k]
    for j in range(len(items)):
        item=items[j];seen=False
        for previous in range(j):
            if items[previous]==item:seen=True
        if seen:continue
        error=0.;count=0
        for q in range(len(items)):
            if items[q]==item:error+=errors[q];count+=1
        for k in range(d):
            g=error*z[k]+count*reg*V[item,k]
            av[item,k]+=g*g;V[item,k]-=lr*g/math.sqrt(av[item,k])
        for value,acc,index in [(B,aB,bucket),(T,aT,kind)]:
            g=error+count*ctxreg*value[item,index]
            acc[item,index]+=g*g;value[item,index]-=lr*g/math.sqrt(acc[item,index])
        if event_bias:
            g=error+count*reg*bi[item];abi[item]+=g*g;bi[item]-=lr*g/math.sqrt(abi[item])
    for k in range(d):
        g=gradient[k]+reg*U[u,k];au[u,k]+=g*g;U[u,k]-=lr*g/math.sqrt(au[u,k])
    # Aggregate repeated history IDs once; regularization is per occurrence.
    for q in range(len(history)):
        h=history[q]
        if h<0:continue
        seen=False
        for j in range(q):
            if history[j]==h:seen=True
        if seen:continue
        count=0
        for item in history:
            if item==h:count+=1
        for k in range(d):
            g=count*(scale*gradient[k]+reg*H[h,k])
            ah[h,k]+=g*g;H[h,k]-=lr*g/math.sqrt(ah[h,k])
    if event_bias:
        g=errors.sum()+reg*bu[u];abu[u]+=g*g;bu[u]-=lr*g/math.sqrt(abu[u])


@njit
def sequence_epoch(order,users,positive,history,buckets,types,negatives,
                   U,V,H,bu,bi,B,T,au,av,ah,abu,abi,aB,aT,lr,reg,ctxreg,weight,pop_bias):
    loss=0.
    for q in order:
        u=users[q];b=buckets[q];t=types[q]
        items=np.empty(negatives.shape[1]+1,np.int32);items[0]=positive[q];items[1:]=negatives[q]
        z,scale=profile(u,history[q],U,H);logits=np.empty(len(items))
        for j in range(len(items)):
            i=items[j];logits[j]=pop_bias[i]+B[i,b]+T[i,t]+np.dot(z,V[i])
        maximum=logits.max();p=np.exp(logits-maximum);denom=p.sum()
        loss+=math.log(denom)+maximum-logits[0]
        errors=weight*p/denom;errors[0]-=weight
        apply_step(u,items,history[q],b,t,z,scale,errors,U,V,H,bu,bi,B,T,
                   au,av,ah,abu,abi,aB,aT,lr,reg*weight,ctxreg*weight,False)
    return loss/max(len(order),1)


@njit
def contextual_epoch(order,users,items,history,target,buckets,types,
                     U,V,H,bu,bi,B,T,au,av,ah,abu,abi,aB,aT,lr,reg,ctxreg):
    loss=0.
    for e in order:
        u=users[e];b=buckets[e];t=types[e]
        z,scale=profile(u,history[e],U,H);logits=np.empty(10)
        for j in range(10):
            i=items[e,j];logits[j]=-1.8+bu[u]+bi[i]+B[i,b]+T[i,t]+np.dot(z,V[i])
        maximum=max(0.,logits.max());p=np.exp(logits-maximum);denom=math.exp(-maximum)+p.sum()
        loss+=math.log(denom)+maximum-(logits[target[e]] if target[e]<10 else 0.)
        errors=p/denom
        if target[e]<10:errors[target[e]]-=1.
        apply_step(u,items[e],history[e],b,t,z,scale,errors,U,V,H,bu,bi,B,T,
                   au,av,ah,abu,abi,aB,aT,lr,reg,ctxreg,True)
    return loss/max(len(order),1)


@njit
def contextual_predict(users,items,history,buckets,types,U,V,H,bu,bi,B,T):
    result=np.empty((len(users),10),np.float64)
    for e in range(len(users)):
        u=users[e];z,scale=profile(u,history[e],U,H);b=buckets[e];t=types[e]
        logits=np.empty(10)
        for j in range(10):
            i=items[e,j];logits[j]=-1.8+bu[u]+bi[i]+B[i,b]+T[i,t]+np.dot(z,V[i])
        maximum=max(0.,logits.max());p=np.exp(logits-maximum)
        result[e]=p/(math.exp(-maximum)+p.sum())
    return result
