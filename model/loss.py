"""The term called "local" is not local. Swap the two roles so that it is.

As it stands the local term scores 6 s, 9 s, ... 42 s - every one of them measured from the START of the
window. Nothing in the loss ever looks at a stretch on its own. That makes attribution muddy in a way that
matters for the thing being chased here: to decide whether a candidate should have been dropped in one
chunk, the cost of that chunk has to be separable from everything that came before it, and right now it is
smeared across every prefix that contains it.

The user's proposal, implemented here:

    local     the CURRENT window only, non-overlapping, no accumulation
    global    takes over the accumulated ladder, 3 s 6 s ... 42 s

Two readings of "global does it too", both run, because they differ in one thing that matters - how much
weight is left on the full 42 s, which is the length the reported number is computed at:

    S1   local 0.3 (6 s windows) + global 0.6 (ladder 3..42 s)          42 s is one rung of fourteen
    S2   local 0.3 (6 s windows) + ladder 0.3 + a 42 s term at 0.3      42 s keeps a term of its own

against base, which is what the model is trained with today. Weights sum to 0.9 in every arm, the same
total as now, so a difference is the STRUCTURE and not the amount of pressure.

The local window is 6 s, not 3 s. A 3 s stretch has an inflated null - almost any band-limited signal
"matches" one - so a per-3 s correlation would be mostly gradient noise. The ladder does start at 3 s, as
asked, but it is one rung of fourteen there rather than the whole signal.

GUARDRAIL, reported for every arm: the step across a chunk seam against the step inside a chunk. Scoring
each window on its own removes the constraint that neighbouring windows share a scale, and the seam ratio
is currently 1.04x against the belt's own - if it climbs, this structure broke the continuity that the
accumulated form was buying, and no gain elsewhere is worth that.

Zone CE stays at 1.0. It was measured this week: dropping it to 0.1 cost -0.027 |corr| (p=0.0032), so the
term is load-bearing regardless of the zone accuracy it reports.

Start noise is tied to the dataset row, shared across arms, KD draws per window - so the arm comparison is
paired on the noise and the draw lottery cancels out of it.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from scipy.stats import ttest_rel
from metrics import score,zn
import diffusion_model as D
import network as V1
from determinism import set_det,gen_fixed
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
EP=int(os.environ.get('EP','60'));SEED=int(os.environ.get('SEED','0'))
NFOLD=int(os.environ.get('NFOLD','5'));DET=int(os.environ.get('DET','1'))
KD=int(os.environ.get('KD','8'))
LOCW=int(os.environ.get('LOCW','2'))                  # chunks per local window: 2 = 6 s
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N;XIQ=V1.XIQ0
NW=NC//LOCW
# ---- the three loss shapes ------------------------------------------------------------------------
def L_window(x0,gt,B):
    """each non-overlapping LOCW-chunk window on its own. No accumulation, nothing before it."""
    a=x0[:,:NW*LOCW].reshape(B,NW,LOCW*CH);b=gt[:,:NW*LOCW].reshape(B,NW,LOCW*CH)
    return (1-D.rho(a,b)).mean()
def L_ladder(x0,gt,B,q0=0):
    """accumulated prefixes, q0=0 gives 3,6,...,42 s and q0=1 gives 6,...,42 s (today's local)"""
    return sum((1-D.rho(x0[:,:q+1].reshape(B,-1),gt[:,:q+1].reshape(B,-1))).mean()
               for q in range(q0,NC))/(NC-q0)
def L_full(x0,gt,B):
    return (1-D.rho(x0.reshape(B,-1),gt.reshape(B,-1))).mean()
def loss_base(x0,gt,B):   return 0.3*L_ladder(x0,gt,B,q0=1)+0.6*L_full(x0,gt,B)
def loss_S1(x0,gt,B):     return 0.3*L_window(x0,gt,B)+0.6*L_ladder(x0,gt,B,q0=0)
def loss_S2(x0,gt,B):     return 0.3*L_window(x0,gt,B)+0.3*L_ladder(x0,gt,B,q0=0)+0.3*L_full(x0,gt,B)
ARMS=[('base',loss_base),('S1 window + ladder',loss_S1),('S2 window + ladder + 42s',loss_S2)]
PICK=os.environ.get('ARMS')
if PICK: ARMS=[a for a in ARMS if a[0].split()[0] in PICK.split(',')]
# ---- the guardrail --------------------------------------------------------------------------------
BND=np.arange(1,NC)*CH-1                              # the step that crosses each chunk boundary
INS=np.ones(W-1,bool);INS[BND]=False
def seam(A):
    """mean step across a seam / median step inside a chunk. The belt's own value is about 1.0"""
    d=np.abs(np.diff(np.stack([zn(a) for a in A]),axis=1))
    return float(d[:,BND].mean()/(np.median(d[:,INS])+1e-9))
def run_arm(name,lossf):
    subs=np.array([u for u in np.unique(S) if u not in (2,10,13,21)])
    pmx=np.random.RandomState(0).permutation(len(subs));folds=[subs[pmx[i::NFOLD]] for i in range(NFOLD)]
    EV=np.concatenate([np.flatnonzero(S==u)[::7] for u in subs])
    rs=np.random.RandomState(1);out=np.isin(S,[2,10,13,21])
    DON=np.array([rs.choice(np.flatnonzero((S!=S[i])&~out)) for i in range(N)])
    Gt2=G.reshape(N,NC,CH).astype(np.float32)
    OUT=np.zeros((N,KD,W),np.float32);DNR=np.zeros((N,W),np.float32)
    for fi,te in enumerate(folds):
        trm=~np.isin(S,te)&~np.isin(S,[2,10,13,21])
        Cn,Pn=D.fold_norm(trm);T_=lambda a,m:torch.tensor(a[m],device=dev)
        ct,pt,zt,gt,xt=T_(Cn,trm),T_(Pn,trm),T_(ZONE,trm),T_(Gt2,trm),T_(XIQ,trm)
        torch.manual_seed(900+fi+1000*SEED);net=V1.Net().to(dev)
        ema=copy.deepcopy(net);[p.requires_grad_(False) for p in ema.parameters()]
        opt=torch.optim.AdamW(net.parameters(),2e-4,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP);n=len(xt)
        for ep in range(EP):
            net.train();idx=torch.randperm(n,device=dev)
            for b in range(0,n,32):
                j=idx[b:b+32];B=len(j)
                tok,_,lz=net.cond(V1.rotate(xt[j]),ct[j],pt[j])
                t=torch.randint(0,D.T_DIFF,(B,),device=dev);eps=torch.randn_like(gt[j])
                a=D.ab[t][:,None,None];xn=a.sqrt()*gt[j]+(1-a).sqrt()*eps
                e=net(xn,t,tok);l=Fn.mse_loss(e,eps)
                x0=(xn-(1-a).sqrt()*e)/a.sqrt()
                l=l+Fn.cross_entropy(lz.reshape(-1,14),zt[j].reshape(-1),ignore_index=-1)
                l=l+lossf(x0,gt[j],B)
                opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                with torch.no_grad():
                    for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(.999).add_(pn,alpha=.001)
            sch.step()
        ema.eval();tem=EV[np.isin(S[EV],te)]
        with torch.no_grad():
            for b in range(0,len(tem),16):
                kk=tem[b:b+16]
                ci=torch.tensor(Cn[kk],device=dev);pi=torch.tensor(Pn[kk],device=dev)
                xi=torch.tensor(XIQ[kk],device=dev)
                tok,_,_=ema.cond(xi,ci,pi);tku,_,_=ema.cond(torch.zeros_like(xi),ci,pi)
                for d in range(KD):
                    v=gen_fixed(ema,tok,tku,kk,1,seed=7+d) if DET else V1.gen(ema,tok,tku,len(kk),1,seed=7+d)
                    OUT[kk,d]=v.cpu().numpy().reshape(len(kk),-1)
                xd=torch.tensor(XIQ[DON[kk]],device=dev)
                tk2,_,_=ema.cond(xd,ci,pi);tu2,_,_=ema.cond(torch.zeros_like(xd),ci,pi)
                v=gen_fixed(ema,tk2,tu2,kk,1) if DET else V1.gen(ema,tk2,tu2,len(kk),1)
                DNR[kk]=v.cpu().numpy().reshape(len(kk),-1)
        del net,ema;torch.cuda.empty_cache();print(f'    {name}  fold{fi} done',flush=True)
    SE=S[EV];Y=OUT[EV]
    cor=lambda a,b:abs(np.corrcoef(zn(a),zn(b))[0,1])
    ccd=np.array([[cor(Y[j,d],G[i]) for d in range(KD)] for j,i in enumerate(EV)])
    mn=np.array([zn(Y[j]).mean(0) for j in range(len(EV))])
    ccm=np.array([cor(mn[j],G[i]) for j,i in enumerate(EV)])
    dn=np.array([cor(DNR[i],G[i]) for i in EV])
    f1=np.array([score(Y[j,0],G[i],FS)['breath_f1'] for j,i in enumerate(EV)])
    fm=np.array([score(mn[j],G[i],FS)['breath_f1'] for j,i in enumerate(EV)])
    df=np.array([score(DNR[i],G[i],FS)['breath_f1'] for i in EV])
    per=lambda v:np.array([v[SE==u].mean() for u in subs])
    return dict(subs=subs,eval_idx=EV,cc=per(ccd[:,0]),ccd=ccd,ccm=per(ccm),dn=per(dn),
                f1=per(f1),fm=per(fm),df=per(df),cc_win=ccd[:,0],
                seam=seam(Y[:,0]),seam_k=seam(mn),wv=Y[:,0],wvm=mn,wvall=Y.astype(np.float32),dw=DNR[EV])
if __name__=='__main__':
    set_det(DET)
    print(f'  seed {SEED}   EP {EP}   folds {NFOLD}   local window {LOCW*CH/FS:.0f} s ({NW} per 42 s)'
          f'   {KD} start noises')
    print(f'  arms: '+' | '.join(n for n,_ in ARMS),flush=True)
    EVref=np.concatenate([np.flatnonzero(S==u)[::7]
                          for u in np.unique(S) if u not in (2,10,13,21)])
    print(f'  the belt\'s own seam ratio (the number to stay near): {seam(G[EVref]):.3f}\n',flush=True)
    R={}
    for name,lf in ARMS:
        print(f'  --- {name} ---',flush=True);R[name]=run_arm(name,lf)
    print(f'\n  one start noise, the same one in every arm')
    print(f'  {"arm":26s} {"|corr|":>8s} {"donor":>8s} {"real":>8s} {"F1":>8s} {"donor":>8s} {"real":>8s}'
          f' {"seam":>7s}')
    for name,_ in ARMS:
        r=R[name]
        print(f'  {name:26s} {r["cc"].mean():8.4f} {r["dn"].mean():8.4f} '
              f'{r["cc"].mean()-r["dn"].mean():+8.4f} {r["f1"].mean():8.4f} {r["df"].mean():8.4f} '
              f'{r["f1"].mean()-r["df"].mean():+8.4f} {r["seam"]:7.3f}')
    print(f'\n  the {KD} draws averaged')
    for name,_ in ARMS:
        r=R[name];d=r['ccd']
        print(f'  {name:26s} K-mean {r["ccm"].mean():.4f} ({r["ccm"].mean()-r["cc"].mean():+.4f})  '
              f'F1 {r["fm"].mean():.4f}  seam {r["seam_k"]:.3f}  '
              f'draw spread {(d.max(1)-d.min(1)).mean():.4f}')
    if 'base' in R:
        print(f'\n  paired against base over {len(R["base"]["subs"])} subjects')
        for name,_ in ARMS:
            if name=='base': continue
            a,b=R[name]['cc'],R['base']['cc'];fa,fb=R[name]['f1'],R['base']['f1']
            ra=R[name]['cc']-R[name]['dn'];rb=R['base']['cc']-R['base']['dn']
            print(f'  {name:26s} |corr| {a.mean()-b.mean():+.4f} p {ttest_rel(a,b).pvalue:.4f}'
                  f'   real {ra.mean()-rb.mean():+.4f} p {ttest_rel(ra,rb).pvalue:.4f}'
                  f'   F1 {fa.mean()-fb.mean():+.4f} p {ttest_rel(fa,fb).pvalue:.4f}')
    np.savez_compressed(f'{PRE}/_loss_struct{GTAG}_s{SEED}.npz',seed=SEED,det=DET,ep=EP,kd=KD,locw=LOCW,
                        arms=np.array([n for n,_ in ARMS],dtype=object),
                        seam_gt=seam(G[EVref]),
                        subs=R[ARMS[0][0]]['subs'],eval_idx=R[ARMS[0][0]]['eval_idx'],
                        **{f'{k}_{i}':R[n][k] for i,(n,_) in enumerate(ARMS)
                           for k in ('cc','ccd','ccm','dn','f1','fm','df','cc_win','seam','seam_k',
                                     'wv','wvm','wvall','dw')})
    print(f'\n  saved {PRE}/_loss_struct{GTAG}_s{SEED}.npz\n')
