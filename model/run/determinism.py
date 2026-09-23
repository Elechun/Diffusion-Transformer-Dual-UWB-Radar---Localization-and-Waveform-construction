"""Two runs of the same thing should give the same number. Right now they do not, and that has to be fixed
before any loss change can be read.

Cohort scores move +-0.021 and per-subject scores +-0.084 between runs of identical weights, which is larger
than every effect we have been arguing about. Until that is closed, a loss arm that comes in at +0.02 cannot
be told from the same arm run twice.

The wobble has two possible homes and they need different fixes, so they are measured apart:

    inference   the SAME trained weights, sampled twice. Everything here is our own choice - the DDIM start
                noise, and whether the cuda kernels are allowed to pick a fast non-reproducible algorithm.
    training    the same seed, trained twice from scratch. If the weights themselves come out different,
                fixing the sampler will not save us.

Two arms:

    as-is       exactly what the current scripts do: torch.manual_seed(7) once per generation BATCH, so a
                row's start noise depends on where it landed in the batch, and cudnn free to choose
    fixed       every eval row draws its start noise from a generator seeded by its own dataset index, so
                batching cannot touch it, plus deterministic algorithms and cudnn.deterministic

Cheap on purpose - one fold, few epochs. It is not measuring how good the model is, only whether the model
says the same thing twice.
"""
import _path  # noqa: F401  (see _path.py)
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')      # must precede the cuda context
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from normalize import zn
import diffusion_model as D
import network as V1
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
EP=int(os.environ.get('EP','12'));SEED=int(os.environ.get('SEED','0'))
DET=int(os.environ.get('DET','0'))
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N;XIQ=V1.XIQ0
def set_det(on):
    torch.backends.cudnn.deterministic=bool(on);torch.backends.cudnn.benchmark=not on
    torch.use_deterministic_algorithms(bool(on),warn_only=True)
def start_noise(rows,nchan,seed=7):
    """DDIM start noise tied to each row's dataset index, so batch size and order cannot change it"""
    out=torch.empty(len(rows),NC,nchan*CH,device=dev)
    for i,r in enumerate(rows):
        g=torch.Generator(device='cpu');g.manual_seed(1000003*seed+int(r))
        out[i]=torch.randn(NC,nchan*CH,generator=g).to(dev)
    return out
@torch.no_grad()
def gen_fixed(net,tok,tku,rows,nchan,seed=7,steps=50):
    """V1.gen with the start noise handed in rather than drawn from the global rng"""
    n=len(rows);x=start_noise(rows,nchan,seed)
    ts=torch.linspace(D.T_DIFF-1,0,steps).long().to(dev)
    for i in range(steps):
        t=ts[i].repeat(n);ec=net(x,t,tok);eu=net(x,t,tku);e=eu+V1.GS*(ec-eu)
        a=D.ab[ts[i]];x0=V1.band_any((x-(1-a).sqrt()*e)/a.sqrt(),nchan)
        x0c=V1.band_any((x-(1-a).sqrt()*ec)/a.sqrt(),nchan)
        r=(x0c.reshape(n,-1).std(1)/(x0.reshape(n,-1).std(1)+1e-9)).reshape(n,1,1)
        x0=V1.PHI*(x0*r)+(1-V1.PHI)*x0;e=(x-a.sqrt()*x0)/(1-a).sqrt()
        if i<steps-1: a2=D.ab[ts[i+1]];x=a2.sqrt()*x0+(1-a2).sqrt()*e
        else: x=x0
    return x
def train_fold(te,trm,seed,Cn,Pn,Gt2,ct,pt,zt,gt,xt):
    torch.manual_seed(900+1000*seed);net=V1.Net().to(dev)
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
            l=l+0.3*sum((1-D.rho(x0[:,:q+1].reshape(B,-1),gt[j][:,:q+1].reshape(B,-1))).mean()
                        for q in range(1,NC))/(NC-1)
            l=l+0.6*(1-D.rho(x0.reshape(B,-1),gt[j].reshape(B,-1))).mean()
            opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():
                for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(.999).add_(pn,alpha=.001)
        sch.step()
    ema.eval();return ema
def emit(ema,rows,Cn,Pn,fixed):
    out=np.zeros((len(rows),W),np.float32)
    with torch.no_grad():
        for b in range(0,len(rows),16):
            # index with kk, never a boolean mask: a mask silently re-sorts and would pair each window's
            # geometry with a different window's candidates the moment the rows are not already ascending
            kk=rows[b:b+16]
            ci=torch.tensor(Cn[kk],device=dev);pi=torch.tensor(Pn[kk],device=dev)
            xi=torch.tensor(XIQ[kk],device=dev)
            tok,_,_=ema.cond(xi,ci,pi);tku,_,_=ema.cond(torch.zeros_like(xi),ci,pi)
            v=gen_fixed(ema,tok,tku,kk,1) if fixed else V1.gen(ema,tok,tku,len(kk),1)
            out[b:b+len(kk)]=v.cpu().numpy().reshape(len(kk),-1)
    return out
def cc(out,rows): return np.array([abs(np.corrcoef(zn(out[i]),zn(G[rows[i]]))[0,1]) for i in range(len(rows))])
if __name__=='__main__':
    set_det(DET)
    print(f'  arm "{"fixed" if DET else "as-is"}"   EP {EP}   seed {SEED}   '
          f'deterministic_algorithms {bool(DET)}\n',flush=True)
    subs=np.array([u for u in np.unique(S) if u not in (2,10,13,21)])
    pmx=np.random.RandomState(0).permutation(len(subs));te=subs[pmx[0::5]]
    trm=~np.isin(S,te)&~np.isin(S,[2,10,13,21])
    Cn,Pn=D.fold_norm(trm);Gt2=G.reshape(N,NC,CH).astype(np.float32)
    T_=lambda a,m:torch.tensor(a[m],device=dev)
    ct,pt,zt,gt,xt=T_(Cn,trm),T_(Pn,trm),T_(ZONE,trm),T_(Gt2,trm),T_(XIQ,trm)
    rows=np.concatenate([np.flatnonzero(S==u)[::7] for u in te])
    print(f'  fold 0: {trm.sum()} train windows, {len(rows)} eval windows over {len(te)} subjects',flush=True)
    ema=train_fold(te,trm,SEED,Cn,Pn,Gt2,ct,pt,zt,gt,xt);print('  train 1 done',flush=True)
    a1=emit(ema,rows,Cn,Pn,DET);a2=emit(ema,rows,Cn,Pn,DET)                 # same weights, twice
    # a third pass with the eval rows in a different batching, to catch batch-position noise
    perm=np.random.RandomState(5).permutation(len(rows));a3=emit(ema,rows[perm],Cn,Pn,DET)
    inv=np.empty_like(perm);inv[perm]=np.arange(len(perm));a3=a3[inv]
    w1={k:v.detach().clone() for k,v in ema.state_dict().items()}
    del ema;torch.cuda.empty_cache()
    ema2=train_fold(te,trm,SEED,Cn,Pn,Gt2,ct,pt,zt,gt,xt);print('  train 2 done',flush=True)
    b1=emit(ema2,rows,Cn,Pn,DET)
    wdiff=max(float((w1[k]-v).abs().max()) for k,v in ema2.state_dict().items() if v.dtype.is_floating_point)
    def rep(tag,p,q):
        c1,c2=cc(p,rows),cc(q,rows)
        wf=float(np.abs(p-q).max())
        ps=np.array([np.mean(c1[np.isin(S[rows],[u])])-np.mean(c2[np.isin(S[rows],[u])]) for u in te])
        print(f'  {tag:34s} waveform max diff {wf:9.2e}   cohort |corr| {c1.mean():.4f} vs {c2.mean():.4f} '
              f'= {c1.mean()-c2.mean():+.4f}   per-subject |diff| max {np.abs(ps).max():.4f}')
    print()
    rep('same weights, sampled twice',a1,a2)
    rep('same weights, rows re-batched',a1,a3)
    rep('same seed, trained twice',a1,b1)
    print(f'\n  largest weight difference between the two trainings: {wdiff:.3e}')
    np.savez_compressed(f'{PRE}/_determinism{GTAG}_det{DET}_s{SEED}.npz',rows=rows,a1=a1,a2=a2,a3=a3,b1=b1,
                        wdiff=wdiff,ep=EP,det=DET)
    print(f'  saved {PRE}/_determinism{GTAG}_det{DET}_s{SEED}.npz\n')
