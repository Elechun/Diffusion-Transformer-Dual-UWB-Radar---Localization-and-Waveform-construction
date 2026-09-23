"""Rate control at sampling time: bend each denoising step toward the calibration rate's band.

The pacer failed for an identifiable reason: a fixed-phase sine has zero expected correlation with the
belt during training, so the model was right to ignore it, and every inference variant came out the same.
This attempt does not ask the model to learn anything. The DDIM loop already band-limits x0 to 0.1-0.6 Hz
at every step; here that projection is partially replaced by a narrow gaussian band centred on the rate
the calibration block supplies. The generator cannot ignore it - it is inside the sampler - and the phase
inside the passband is the model's own, so the radar's timing information is kept.

    x0  <-  (1-g) * wideband(x0)  +  g * narrowband(x0; r, sigma)

One envdphi_s2 model per fold, trained exactly as before; five generation variants on the same weights:

    none      g=0, the arm as it was
    calib     g=G, r from the subject's first 84 s          - the deployable arm
    oracle    g=G, r from the whole-session belt            - the upper bound
    const20   g=G, r=20 for everyone                        - must look like the old collapse
    shuffled  g=G, r from another subject's calibration     - must be worse than calib

Same leakage rule as before: eval windows never overlap the first 84 s.
"""
import _path  # noqa: F401  (see _path.py)
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from scipy.stats import pearsonr
from normalize import zn
from metrics import score
import diffusion_model as D
import network as V1
import loss as LS
import augment as RA
import rate_condition as RC2
import metrics_windowed as T
from determinism import set_det,start_noise
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N
EXCL=tuple(int(v) for v in os.environ.get('EXCL','2,10,13,21,22').split(','))
SEED=int(os.environ.get('SEED','0'));EP=int(os.environ.get('EP','60'));NFOLD=5
GUIDE=float(os.environ.get('GUIDE','0.4'));SIG=float(os.environ.get('SIG','1.2'))   # bpm
VARIANTS=['none','calib','oracle','const20','shuffled']

_FR=np.fft.rfftfreq(W,1/FS)*60.0
def narrow_mask(r_bpm):
    m=np.exp(-((_FR-r_bpm)**2)/(2*SIG**2))
    return torch.tensor(m,device=dev,dtype=torch.float32)


GSCHED=os.environ.get('GSCHED','')   # step schedule for the narrow-band rate guidance: 'a:b:k' = g=a for DDIM steps i<k, g=b afterwards; 'lin:a:b' = linear a->b over the steps; '' = constant g (base)
def gstep(g,i,steps):
    """guidance strength at DDIM step i (0 = noisiest); falls back to the constant g when GSCHED is unset"""
    if not GSCHED: return g
    p=GSCHED.split(':')
    if p[0]=='lin': a,b=float(p[1]),float(p[2]);return a+(b-a)*i/max(1,steps-1)
    a,b,k=float(p[0]),float(p[1]),int(p[2]);return a if i<k else b

@torch.no_grad()
def gen_guided(net,tok,tku,rows,rates,g,seed=7,steps=50):
    """gen_fixed with the extra narrowband blend on x0; rates per row, g=0 reproduces the original"""
    n=len(rows);x=start_noise(rows,1,seed)
    masks=torch.stack([narrow_mask(float(r)) for r in rates])          # (n,F)
    ts=torch.linspace(D.T_DIFF-1,0,steps).long().to(dev)
    for i in range(steps):
        t=ts[i].repeat(n);ec=net(x,t,tok);eu=net(x,t,tku);e=eu+V1.GS*(ec-eu)
        a=D.ab[ts[i]];x0=V1.band_any((x-(1-a).sqrt()*e)/a.sqrt(),1)
        x0c=V1.band_any((x-(1-a).sqrt()*ec)/a.sqrt(),1)
        r_=(x0c.reshape(n,-1).std(1)/(x0.reshape(n,-1).std(1)+1e-9)).reshape(n,1,1)
        x0=V1.PHI*(x0*r_)+(1-V1.PHI)*x0
        gi=gstep(g,i,steps)
        if gi>0:
            v=x0.reshape(n,-1)
            Fv=torch.fft.rfft(v,dim=1)
            nb=torch.fft.irfft(Fv*masks,n=W,dim=1)
            v=(1-gi)*v+gi*nb
            x0=v.reshape(n,NC,CH)
        e=(x-a.sqrt()*x0)/(1-a).sqrt()
        if i<steps-1: a2=D.ab[ts[i+1]];x=a2.sqrt()*x0+(1-a2).sqrt()*e
        else: x=x0
    return x

if __name__=='__main__':
    set_det(1)
    subs=np.array([u for u in np.unique(S) if u not in EXCL])
    pmx=np.random.RandomState(0).permutation(len(subs))
    folds=[subs[pmx[i::NFOLD]] for i in range(NFOLD)]
    ED=RA.envdphi_cache()
    whole={};calib={}
    for u in np.unique(S):
        gg=np.load(f'{PRE}/sub{u:02d}/data_fc729.npz',allow_pickle=True)['gt_aligned'].astype(float)
        gg=gg[np.isfinite(gg)]
        whole[u]=T.bpm(zn(gg));calib[u]=T.bpm(zn(gg[:int(84*FS)]))
    rs=np.random.RandomState(9)
    partner={u:int(rs.choice([v for v in subs if v!=u])) for u in subs}
    Gt2=G.reshape(N,NC,CH).astype(np.float32)
    OUT={v:np.zeros((N,W),np.float32) for v in VARIANTS}
    print(f'  seed {SEED}  EP {EP}  guide g={GUIDE} sigma={SIG} bpm   envdphi_s2 weights',flush=True)
    for fi,te in enumerate(folds):
        trm=~np.isin(S,te)&~np.isin(S,EXCL)
        Cn,Pn=D.fold_norm(trm);T_=lambda a,m:torch.tensor(a[m],device=dev)
        ct,pt,zt,gt,xt=T_(Cn,trm),T_(Pn,trm),T_(ZONE,trm),T_(Gt2,trm),T_(ED,trm)
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
            rows=RC2.evrows(u)
            if not len(rows): continue
            rmap={'none':None,'calib':calib[u],'oracle':whole[u],'const20':20.0,
                  'shuffled':calib[partner[int(u)]]}
            for v in VARIANTS:
                for b in range(0,len(rows),16):
                    kk=rows[b:b+16]
                    ci=torch.tensor(Cn[kk],device=dev);pi=torch.tensor(Pn[kk],device=dev)
                    xi=torch.tensor(ED[kk],device=dev)
                    tok,_,_=ema.cond(xi,ci,pi);tku,_,_=ema.cond(torch.zeros_like(xi),ci,pi)
                    if v=='none':
                        OUT[v][kk]=gen_guided(ema,tok,tku,kk,[20.0]*len(kk),0.0).cpu().numpy().reshape(len(kk),-1)
                    else:
                        OUT[v][kk]=gen_guided(ema,tok,tku,kk,[rmap[v]]*len(kk),GUIDE).cpu().numpy().reshape(len(kk),-1)
        del net,ema;torch.cuda.empty_cache();print(f'    fold{fi} done',flush=True)

    allrows=np.concatenate([RC2.evrows(u) for u in subs])
    SE=S[allrows];BEL=np.stack([zn(G[i]) for i in allrows])
    gtr=np.array([T.bpm(BEL[j]) for j in range(len(allrows))])
    print(f'\n  {"variant":9s} {"corr":>7s} {"F1":>7s} {"rate sd":>8s} {"r(belt)":>8s} {"MAE":>6s}')
    for v in VARIANTS:
        REC=np.stack([zn(OUT[v][i]) for i in allrows])
        cc=np.mean([np.corrcoef(REC[j],BEL[j])[0,1] for j in range(len(allrows))])
        f1=np.mean([score(REC[j],BEL[j],FS)['breath_f1'] for j in range(len(allrows))])
        pr=np.array([T.bpm(REC[j]) for j in range(len(allrows))])
        ps=np.array([pr[SE==u].mean() for u in subs]);gs=np.array([gtr[SE==u].mean() for u in subs])
        r,p=pearsonr(ps,gs)
        print(f'  {v:9s} {cc:+7.3f} {f1:7.3f} {ps.std():8.2f} {r:+8.3f} {np.abs(ps-gs).mean():6.2f}'
              f'   p={p:.2e}')
    np.savez_compressed(f'{PRE}/_specguide{GTAG}_g{GUIDE}_s{SEED}.npz',seed=SEED,guide=GUIDE,sig=SIG,
                        subs=subs,eval_rows=allrows,
                        **{f'wv_{v}':OUT[v][allrows] for v in VARIANTS})
    print(f'\n  saved {PRE}/_specguide{GTAG}_g{GUIDE}_s{SEED}.npz')
