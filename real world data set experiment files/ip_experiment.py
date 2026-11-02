"""Exact integer-programming (MILP) vs greedy+local-search for equal-size,
skill-balanced, socially cohesive study groups. Real data: UCI Student Performance (Portuguese)."""
import itertools, json, time, sys
import numpy as np, pandas as pd

N, K = 15, 3            # 15 students, 3 groups of 5 (exact MILP tractable)
ALPHA = 0.5
REPS = 20
SEED = 20261024
TL = 120                # solver time limit (s)

# Locate dataset whether run from root or from within the folder
from pathlib import Path
_base = Path(__file__).resolve().parent
_candidates = [
    _base / "student-por.csv",
    _base.parent / "student+performance" / "student-por.csv",
    Path("student+performance/student-por.csv"),
    Path("student-por.csv"),
]
_data_path = next((p for p in _candidates if p.exists()), Path("student-por.csv"))
df = pd.read_csv(_data_path, sep=None, engine="python")

def skills(d):
    a = pd.DataFrame({
        "academic":   d.G1/20,                       # prior grade (period 1)
        "effort":     (d.studytime-1)/3,             # weekly study time
        "attendance": 1-np.minimum(d.absences,30)/30})
    return a.values

def social(d):
    """Pairwise compatibility from real behavioural/lifestyle attributes (derived, not observed friendships)."""
    n=len(d); W=np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        a,b=d.iloc[i],d.iloc[j]
        s =(a.school==b.school)*0.25+(a.address==b.address)*0.10
        s+=(a.sex==b.sex)*0.10+(abs(a.age-b.age)<=1)*0.15
        s+=(1-abs(a.goout-b.goout)/4)*0.20+(1-abs(a.freetime-b.freetime)/4)*0.10
        s+=(a.reason==b.reason)*0.10
        W[i,j]=W[j,i]=s if s>=0.55 else 0.0          # keep only reasonably compatible pairs
    return W

def metrics(g,W,A):
    n=len(g); same=(g[:,None]==g[None,:]); iu=np.triu_indices(n,1)
    C=(W[iu]*same[iu]).sum()/max(W[iu].sum(),1e-12)
    mu=A.mean(0); tot=((A-mu)**2).sum()
    B=sum((g==k).sum()*((A[g==k].mean(0)-mu)**2).sum() for k in set(g))/max(tot,1e-12)
    return C,B

def J(g,W,A,al=ALPHA):
    C,B=metrics(g,W,A); return al*C-(1-al)*B

def greedy_ls(W,A,al=ALPHA,rng=None):
    n=len(W); cap=n//K; g=-np.ones(n,int); cnt=np.zeros(K,int)
    order=np.argsort(-W.sum(1)) 
    for i in order:
        best=None
        for k in range(K):
            if cnt[k]<cap:
                g[i]=k; v=J(np.where(g<0,-1,g),W,A,al) if False else partial(g,W,A,al); 
                if best is None or v>best[0]: best=(v,k)
        g[i]=best[1]; cnt[best[1]]+=1
    imp=True
    while imp:
        imp=False; cur=J(g,W,A,al); bs=None
        for i,j in itertools.combinations(range(n),2):
            if g[i]!=g[j]:
                g[i],g[j]=g[j],g[i]; v=J(g,W,A,al); g[i],g[j]=g[j],g[i]
                if v>cur+1e-12 and (bs is None or v>bs[0]): bs=(v,i,j)
        if bs: i,j=bs[1:]; g[i],g[j]=g[j],g[i]; imp=True
    return g

def partial(g,W,A,al):
    m=g>=0
    if m.sum()<2: return 0
    gg=g[m]; Wm=W[np.ix_(m,m)]; Am=A[m]
    C,B=metrics(gg,Wm,Am); return al*C-(1-al)*B

def solve_ip(W,A,al=ALPHA,tl=TL):
    """Exact MILP (HiGHS). x_ik binary assignment, z_ij edge-retained indicator,
    d_kt >= |sum_i a_it x_ik - (n/K) mu_t| (L1 linearisation of the between-group variance)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    n=len(W); T=A.shape[1]; cap=n//K; mu=A.mean(0); tot=((A-mu)**2).sum(); Wtot=np.triu(W,1).sum()
    edges=[(i,j) for i,j in itertools.combinations(range(n),2) if W[i,j]>0]; E=len(edges)
    nx=n*K; nz=E; nd=K*T; nv=nx+nz+nd
    X=lambda i,k:i*K+k; Z=lambda e:nx+e; D=lambda k,t:nx+nz+k*T+t
    scale=1.0/(cap*np.sqrt(tot/n)*T*K)
    c=np.zeros(nv)
    for e,(i,j) in enumerate(edges): c[Z(e)]=-al*W[i,j]/Wtot
    for k in range(K):
        for t in range(T): c[D(k,t)]=(1-al)*scale
    rows=[];lo=[];hi=[]
    M=lil_matrix((n+K+2*E*K+2*K*T+1,nv)); r=0
    for i in range(n):
        for k in range(K): M[r,X(i,k)]=1
        lo.append(1);hi.append(1);r+=1
    for k in range(K):
        for i in range(n): M[r,X(i,k)]=1
        lo.append(cap);hi.append(cap);r+=1
    for e,(i,j) in enumerate(edges):
        for k in range(K):
            M[r,Z(e)]=1;M[r,X(i,k)]=1;M[r,X(j,k)]=-1;lo.append(-np.inf);hi.append(1);r+=1   # z<=1-xi+xj
            M[r,Z(e)]=1;M[r,X(i,k)]=-1;M[r,X(j,k)]=1;lo.append(-np.inf);hi.append(1);r+=1   # z<=1+xi-xj
    for k in range(K):
        for t in range(T):
            for sgn in (1,-1):
                M[r,D(k,t)]=1
                for i in range(n): M[r,X(i,k)]=-sgn*A[i,t]
                lo.append(-sgn*cap*mu[t]);hi.append(np.inf);r+=1
    M[r,X(0,0)]=1;lo.append(1);hi.append(1);r+=1       # symmetry breaking
    M=M[:r].tocsr()
    integrality=np.concatenate([np.ones(nx),np.zeros(nz+nd)])
    ub=np.concatenate([np.ones(nx+nz),np.full(nd,np.inf)])
    t0=time.time()
    res=milp(c,constraints=LinearConstraint(M,lo,hi),integrality=integrality,bounds=Bounds(np.zeros(nv),ub),
             options={"time_limit":tl,"mip_rel_gap":1e-6})
    rt=time.time()-t0
    if res.x is None: return None,rt,"no_solution"
    g=np.array([int(np.argmax(res.x[X(i,0):X(i,0)+K])) for i in range(n)])
    return g,rt,{0:"optimal",1:"time_limit"}.get(res.status,str(res.status))

def rand_part(n,rng):
    g=np.repeat(np.arange(K),n//K); rng.shuffle(g); return g

if __name__=="__main__":
    rows=[]; rng=np.random.default_rng(SEED)
    for r in range(REPS):
        idx=rng.choice(len(df),N,replace=False); d=df.iloc[idx].reset_index(drop=True)
        A=skills(d); W=social(d)
        if W.sum()==0: continue
        res={}
        t=time.time(); gg=greedy_ls(W,A); res["Greedy+LS"]=(gg,time.time()-t)
        gi,rt,st=solve_ip(W,A); res["IP (HiGHS, L1)"]=(gi,rt)
        rr=[metrics(rand_part(N,rng),W,A) for _ in range(20)]
        rows.append(dict(rep=r,method="Random",C=np.mean([a for a,_ in rr]),B=np.mean([b for _,b in rr]),
                         J=ALPHA*np.mean([a for a,_ in rr])-(1-ALPHA)*np.mean([b for _,b in rr]),runtime=0,status=""))
        for m,(g,rt_) in res.items():
            C,B=metrics(g,W,A); rows.append(dict(rep=r,method=m,C=C,B=B,J=ALPHA*C-(1-ALPHA)*B,runtime=rt_,status=st if "IP" in m else ""))
        print(r,[ (m,round(metrics(g,W,A)[0],3),round(metrics(g,W,A)[1],4)) for m,(g,_) in res.items()],st,flush=True)
    out=pd.DataFrame(rows); out.to_csv("results_by_instance.csv",index=False)
    s=out.groupby("method")[["C","B","J","runtime"]].agg(["mean","std"]).round(4); print(s); s.to_csv("results_summary.csv")
