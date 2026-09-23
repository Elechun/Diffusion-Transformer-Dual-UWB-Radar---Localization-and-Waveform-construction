"""Carry the breathing rate into the generation as a pacer channel, and pay for it with calibration.

The rate is not in the radar - settled four independent ways - but the capacity test showed the
architecture reproduces a rate that arrives through the conditioning (r=+0.83). And an 84 s calibration
block predicts the subject's whole-session rate at r=+0.83, MAE 1.39 bpm. So the deployable design is:
let the calibration supply the number, let a pacer channel deliver it, and let the radar keep the job it
can actually do - phase and event timing.

Channels: the envdphi representation of the three strongest candidates (COM-LOS, COM-Ghost, TV-LOS), and
in the fourth pair's place a pacer [sin 2*pi*r*t, cos 2*pi*r*t]. Training builds the pacer from the belt
rate of the window itself - ordinary conditioning supervision on training subjects. Evaluation never
touches the test subject's ground truth beyond the declared calibration block.

One trained model per fold, four inference variants that differ only in where the pacer's r comes from:

    oracle    the whole-session belt rate of the subject - the upper bound of this design
    calib     the rate of the subject's first 84 s - the deployable arm, the one that answers the user
    const20   r fixed at 20 for everyone - must reproduce the flat-20 output, or the pacer is not in charge
    shuffled  another subject's calibration rate - must be worse than calib, or the pacer is a bystander

Eval windows exclude anything overlapping the first 84 s, the same span-intersection rule the few-shot
experiment used, so the calibration block never scores itself.
"""
import _path  # noqa: F401  (see _path.py)
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from scipy.stats import pearsonr,ttest_rel
from normalize import zn
from metrics import score
import diffusion_model as D
import network as V1
import loss as LS
import augment as RA
import metrics_windowed as T
from determinism import set_det,gen_fixed
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N
EXCL=tuple(int(v) for v in os.environ.get('EXCL','2,10,13,21,22').split(','))
SEED=int(os.environ.get('SEED','0'));EP=int(os.environ.get('EP','60'));NFOLD=5
VARIANTS=['oracle','calib','const20','shuffled']

def pacer(rates_bpm):
    """(n,) bpm -> (n,2,NC,CH) sin/cos pair, z-scored like every other channel"""
    t=np.arange(W)/FS
    ph=2*np.pi*(np.asarray(rates_bpm,float)[:,None]/60.0)*t[None,:]
    out=np.stack([np.sin(ph),np.cos(ph)],1)
    out=(out-out.mean(-1,keepdims=True))/(out.std(-1,keepdims=True)+1e-9)
    return out.reshape(len(rates_bpm),2,NC,CH).astype(np.float32)

def evrows(u):
    """held-out rows whose 42 s span shares nothing with the first-84 s calibration block"""
    hop=getattr(D,'HOP_S',6.0);rows=np.flatnonzero(S==u);ev=np.arange(len(rows))[::int(round(42/hop))]
    keep=[i for i in ev if not (hop*i<84.0 and 0.0<hop*i+42)]
    return rows[keep]

def with_pacer(X3,rates_bpm):
    """envdphi channels of candidates 0..2 plus the pacer pair in slot 3"""
    Xp=np.empty((len(X3),8,NC,CH),np.float32)
    Xp[:,:6]=X3;Xp[:,6:]=pacer(rates_bpm)
    return Xp

if __name__=='__main__':
    set_det(1)
    subs=np.array([u for u in np.unique(S) if u not in EXCL])
    pmx=np.random.RandomState(0).permutation(len(subs))
    folds=[subs[pmx[i::NFOLD]] for i in range(NFOLD)]
    ED=RA.envdphi_cache()[:,:6]                       # candidates 0..2 as [env,dphi]
    wr=np.array([T.bpm(zn(G[i])) for i in range(N)])  # per-window belt rate (training pacer)
    # per-subject rates for the inference variants
    whole={};calib={}
    for u in np.unique(S):
        g=np.load(f'{PRE}/sub{u:02d}/data_fc729.npz',allow_pickle=True)['gt_aligned'].astype(float)
        g=g[np.isfinite(g)]
        whole[u]=T.bpm(zn(g));calib[u]=T.bpm(zn(g[:int(84*FS)]))
    rs=np.random.RandomState(9)
    partner={u:int(rs.choice([v for v in subs if v!=u])) for u in subs}
    Gt2=G.reshape(N,NC,CH).astype(np.float32)
    OUT={v:np.zeros((N,W),np.float32) for v in VARIANTS}
    print(f'  seed {SEED}  EP {EP}  envdphi(3 cands) + pacer   {len(subs)} subjects',flush=True)
    for fi,te in enumerate(folds):
        trm=~np.isin(S,te)&~np.isin(S,EXCL)
        Cn,Pn=D.fold_norm(trm)
        XT=with_pacer(ED[trm],wr[trm])
        T_=lambda a,m=None:torch.tensor(a if m is None else a[m],device=dev)
        ct,pt,zt,gt=T_(Cn,trm),T_(Pn,trm),T_(ZONE,trm),T_(Gt2,trm);xt=T_(XT)
        torch.manual_seed(900+fi+1000*SEED);net=V1.Net().to(dev)
        ema=copy.deepcopy(net);[p.requires_grad_(False) for p in ema.parameters()]
        opt=torch.optim.AdamW(net.parameters(),2e-4,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP);n=len(xt)
        for ep in range(EP):
            net.train();idx=torch.randperm(n,device=dev)
            for b in range(0,n,32):
                j=idx[b:b+32];B=len(j)
                tok,_,lz=net.cond(xt[j],ct[j],pt[j])
                t=torch.randint(0,D.T_DIFF,(B,),device=dev);eps=torch.randn_like(gt[j])
                a=D.ab[t][:,None,None];xn=a.sqrt()*gt[j]+(1-a).sqrt()*eps
                e=net(xn,t,tok);l=Fn.mse_loss(e,eps)
                x0=(xn-(1-a).sqrt()*e)/a.sqrt()
                l=l+Fn.cross_entropy(lz.reshape(-1,14),zt[j].reshape(-1),ignore_index=-1)
                l=l+LS.loss_S2(x0,gt[j],B)
                opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                with torch.no_grad():
                    for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(.999).add_(pn,alpha=.001)
            sch.step()
        ema.eval()
        for u in te:
            rows=evrows(u)
            if not len(rows): continue
            rates={'oracle':whole[u],'calib':calib[u],'const20':20.0,
                   'shuffled':calib[partner[int(u)]]}
            for v in VARIANTS:
                XI=with_pacer(ED[rows],np.full(len(rows),rates[v]))
                with torch.no_grad():
                    for b in range(0,len(rows),16):
                        kk=rows[b:b+16]
                        ci=torch.tensor(Cn[kk],device=dev);pi=torch.tensor(Pn[kk],device=dev)
                        xi=torch.tensor(XI[b:b+16],device=dev)
                        tok,_,_=ema.cond(xi,ci,pi);tku,_,_=ema.cond(torch.zeros_like(xi),ci,pi)
                        OUT[v][kk]=gen_fixed(ema,tok,tku,kk,1).cpu().numpy().reshape(len(kk),-1)
        del net,ema;torch.cuda.empty_cache();print(f'    fold{fi} done',flush=True)

    allrows=np.concatenate([evrows(u) for u in subs])
    SE=S[allrows];BEL=np.stack([zn(G[i]) for i in allrows])
    gtr=np.array([T.bpm(BEL[j]) for j in range(len(allrows))])
    print(f'\n  {"variant":9s} {"corr":>7s} {"F1":>7s} {"rate sd":>8s} {"r(belt)":>8s} {"MAE":>6s}')
    res={}
    for v in VARIANTS:
        REC=np.stack([zn(OUT[v][i]) for i in allrows])
        cc=np.mean([np.corrcoef(REC[j],BEL[j])[0,1] for j in range(len(allrows))])
        f1=np.mean([score(REC[j],BEL[j],FS)['breath_f1'] for j in range(len(allrows))])
        pr=np.array([T.bpm(REC[j]) for j in range(len(allrows))])
        # subject-level rate: what the user actually asked for
        ps=np.array([pr[SE==u].mean() for u in subs]);gs=np.array([gtr[SE==u].mean() for u in subs])
        r,p=pearsonr(ps,gs)
        res[v]=dict(ps=ps,gs=gs)
        print(f'  {v:9s} {cc:+7.3f} {f1:7.3f} {ps.std():8.2f} {r:+8.3f} {np.abs(ps-gs).mean():6.2f}'
              f'   p={p:.2e}')
    np.savez_compressed(f'{PRE}/_ratecond{GTAG}_s{SEED}.npz',seed=SEED,subs=subs,eval_rows=allrows,
                        **{f'wv_{v}':OUT[v][allrows] for v in VARIANTS})
    print(f'\n  saved {PRE}/_ratecond{GTAG}_s{SEED}.npz')
