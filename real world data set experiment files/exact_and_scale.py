import itertools, time, numpy as np, pandas as pd
import ip_experiment as e
from ip_experiment import *

def all_partitions(n=15,K=3):
    cap=n//K; rest=list(range(1,n)); out=[]
    for g0 in itertools.combinations(rest,cap-1):
        g0=(0,)+g0; rem=[i for i in range(n) if i not in g0]
        for g1 in itertools.combinations(rem,cap):
            g=np.empty(n,np.int8); g[list(g0)]=0; g[list(g1)]=1
            g[[i for i in rem if i not in g1]]=2; out.append(g)
    return np.array(out)

PART=all_partitions(); print("partitions",len(PART))
OH=np.eye(3,dtype=np.float32)[PART]            # P,n,K

def exhaustive(W,A,al=ALPHA):
    iu=np.triu(W,1); Wt=iu.sum()
    C=np.einsum('pik,ij,pjk->p',OH,iu.astype(np.float32),OH)/Wt
    mu=A.mean(0); tot=((A-mu)**2).sum()
    S=np.einsum('pik,it->pkt',OH,A.astype(np.float32))          # group sums
    cap=A.shape[0]//3
    B=(((S/cap-mu)**2).sum((1,2))*cap)/tot
    Jv=al*C-(1-al)*B; b=int(np.argmax(Jv)); return PART[b].astype(int),Jv[b]

rng=np.random.default_rng(SEED); rows=[]
for r in range(REPS):
    idx=rng.choice(len(df),N,replace=False); d=df.iloc[idx].reset_index(drop=True)
    A=skills(d); W=social(d)
    gg=greedy_ls(W,A); gi,rt,st=solve_ip(W,A); ge,Jo=exhaustive(W,A)
    rr=[metrics(rand_part(N,rng),W,A) for _ in range(20)]
    Jr=np.mean([ALPHA*a-(1-ALPHA)*b for a,b in rr])
    row=dict(rep=r,J_opt=Jo,J_greedy=J(gg,W,A),J_ip=J(gi,W,A),J_random=Jr,ip_runtime=rt)
    row["greedy_gap"]=(Jo-row["J_greedy"])/abs(Jo); row["ip_gap"]=(Jo-row["J_ip"])/abs(Jo)
    for nm,g in (("opt",ge),("greedy",gg),("ip",gi)):
        row["C_"+nm],row["B_"+nm]=metrics(g,W,A)
    rows.append(row)
R=pd.DataFrame(rows); R.to_csv("exact_comparison.csv",index=False)
print(R[["J_opt","J_greedy","J_ip","J_random","greedy_gap","ip_gap","ip_runtime"]].describe().round(4))
print("greedy optimal in",(R.greedy_gap<1e-6).sum(),"/",len(R)," ip optimal in",(R.ip_gap<1e-6).sum(),"/",len(R))
print("mean C/B opt",R.C_opt.mean().round(4),R.B_opt.mean().round(4),"greedy",R.C_greedy.mean().round(4),R.B_greedy.mean().round(4),"ip",R.C_ip.mean().round(4),R.B_ip.mean().round(4))
