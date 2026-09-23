"""What turned the generator into a template emitter? One change at a time from the August pipeline.

That earlier run - same diffusion model, same EMA, the same 0.3*ladder + 0.6*full loss - produced windows
whose shared template held 21 to 29% of the variance and whose breathing rate had a spread of about 3.2,
close to the belt's 3.7. The current pipeline reaches 64% and 0.25. Since the model and the loss family
are the same, the cause is one of the things that changed since: the BR pre-compensation of the candidates,
the loss reshuffle to S2, or the random I/Q rotation. Each arm below moves exactly one of them.

The finding this answers: 64% of every reconstructed window's variance is one shared template, phase-
locked to the window frame (start-phase concentration 0.90), identical across subjects (+0.85) and
identical under a donor's radar (+0.996). The belt's own window-locked component is 4%. The generator has
collapsed onto E[belt | position-in-window] and lets the radar only bend it - which is exactly what the
user described as "one waveform, slightly deformed, forced to fit".

The suspected cause is the pairing of in-window position with breathing phase: chunk-position embeddings
and the per-chunk geometry context tell the model where in the window it is, the belt has a weak position-
locked mean, and with little phase certainty from the radar the loss is best served by emitting that mean.

The intervention: during training, roll each batch - candidates, belt, geometry context and zone labels
together - by a random whole number of 3 s chunks. The radar-belt pairing is untouched; only the
correspondence between absolute window position and breathing phase is destroyed, so a positional template
stops paying. Evaluation is unrolled and unchanged.

Two arms, one seed to start: S2 as it is, and S2 + roll. Read three things:

    template RMS      does the stack mean collapse toward the belt's own 0.19
    corr / F1         does the freed capacity go into following the belt
    rate sd, r(belt)  does the 20.00 fixation (the window harmonic k=14) release

A wrap seam is created at the roll point of the belt target; it is the same seam in the radar, it moves
every step, and the loss terms are correlations rather than derivatives, so it acts as noise rather than
structure. If the roll arm collapses outright that diagnosis changes.
"""
import _path  # noqa: F401  (see _path.py)
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from scipy.stats import ttest_rel,pearsonr
from scipy.signal import hilbert
from normalize import zn
from metrics import score
import diffusion_model as D
import network as V1
import loss as LS
import br_candidates as BA

def envdphi_cache():
    """Each candidate as [envelope, phase increment] instead of [I, Q].

    Both channels are invariant to a global I/Q rotation - the envelope trivially, the phase increment
    because angle(z e^{jt} * conj(z_prev e^{jt})) drops the common angle - so the arbitrary-axis problem
    the random rotation was built to handle disappears at the representation level, and nothing about the
    breathing phase has to be erased to get there. The increment is wrapped to (-pi, pi] and both channels
    are z-scored per window per candidate, matching how the I/Q pairs were scaled.
    """
    import os as _os
    f=f'{PRE}/_iq_candidates{GTAG}_envdphi.npy'
    if not _os.path.exists(f):
        X=V1.XIQ0.astype(np.float64);Y=np.empty_like(X,dtype=np.float32)
        zn_=lambda v:(v-v.mean(-1,keepdims=True))/(v.std(-1,keepdims=True)+1e-9)
        for c in range(4):
            z=(X[:,2*c]+1j*X[:,2*c+1]).reshape(len(X),-1)
            env=np.abs(z)
            dph=np.angle(z[:,1:]*np.conj(z[:,:-1]))
            dph=np.concatenate([dph[:,:1],dph],1)
            Y[:,2*c]=zn_(env).reshape(-1,NC,CH).astype(np.float32)
            Y[:,2*c+1]=zn_(dph).reshape(-1,NC,CH).astype(np.float32)
        np.save(f,Y);print(f'  built {f}')
    return np.load(f).astype(np.float32)
import metrics_windowed as T
from determinism import set_det,gen_fixed
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N
EXCL=tuple(int(v) for v in os.environ.get('EXCL','2,10,13,21,22').split(','))
SEED=int(os.environ.get('SEED','0'));EP=int(os.environ.get('EP','60'));NFOLD=5
# name -> (candidate source, loss, rotation, roll)
CFG={'aug_base' :('raw',  LS.loss_base,True, False),   # the August configuration
     'br'       :('chunk',LS.loss_base,True, False),   # + BR pre-compensation only
     's2'       :('raw',  LS.loss_S2,  True, False),   # + the S2 loss only
     'norot'    :('raw',  LS.loss_base,False,False),   # - the random I/Q rotation only
     'current'  :('chunk',LS.loss_S2,  True, False),   # everything, i.e. today's baseline
     'roll'     :('chunk',LS.loss_S2,  True, True ),   # today's baseline + the frame-decoupling roll
     'br_norot' :('chunk',LS.loss_base,False,False),
     'brw_norot':('window',LS.loss_base,False,False),
     'envdphi'  :('envdphi',LS.loss_base,False,False),  # rotation-invariant phase representation;
     'envdphi_s2':('envdphi',LS.loss_S2, False,False)}  # rot must stay off: rotate() would mix the
                                                        # two channels as if they were I and Q  # one BR axis per 42 s window: the per-chunk
                                                       # axis flips sign on ~48% of adjacent chunks,
                                                       # which is the suspected source of the in-band
                                                       # irregularity the per-chunk variant shows  # BR fixes the axis deterministically, so the
                                                      # random rotation - now identified as what erases
                                                      # the phase - can be dropped without leaving the
                                                      # axis arbitrary. The user's 'picked + BR'.
ARMS=os.environ.get('ARMS','aug_base,br,s2,norot,current,roll').split(',')

def train_and_eval(mode,SRC,folds,EV):
    src,lossf,rot,roll=CFG[mode];XIQ=SRC[src]
    Gt2=G.reshape(N,NC,CH).astype(np.float32)
    OUT=np.zeros((N,W),np.float32);DNR=np.zeros((N,W),np.float32)
    rs=np.random.RandomState(1);out=np.isin(S,EXCL)
    DON=np.array([rs.choice(np.flatnonzero((S!=S[i])&~out)) for i in range(N)])
    for fi,te in enumerate(folds):
        trm=~np.isin(S,te)&~np.isin(S,EXCL)
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
                xb,cb,zb,gb=xt[j],ct[j],zt[j],gt[j]
                if roll:
                    k=int(torch.randint(0,NC,(1,)).item())
                    if k:
                        xb=torch.roll(xb,k,dims=2);gb=torch.roll(gb,k,dims=1)
                        cb=torch.roll(cb,k,dims=1);zb=torch.roll(zb,k,dims=1)
                xin=V1.rotate(xb) if rot else xb
                tok,_,lz=net.cond(xin,cb,pt[j])
                t=torch.randint(0,D.T_DIFF,(B,),device=dev);eps=torch.randn_like(gb)
                a=D.ab[t][:,None,None];xn=a.sqrt()*gb+(1-a).sqrt()*eps
                e=net(xn,t,tok);l=Fn.mse_loss(e,eps)
                x0=(xn-(1-a).sqrt()*e)/a.sqrt()
                l=l+Fn.cross_entropy(lz.reshape(-1,14),zb.reshape(-1),ignore_index=-1)
                l=l+lossf(x0,gb,B)
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
                OUT[kk]=gen_fixed(ema,tok,tku,kk,1).cpu().numpy().reshape(len(kk),-1)
                xd=torch.tensor(XIQ[DON[kk]],device=dev)
                t2,_,_=ema.cond(xd,ci,pi);u2,_,_=ema.cond(torch.zeros_like(xd),ci,pi)
                DNR[kk]=gen_fixed(ema,t2,u2,kk,1).cpu().numpy().reshape(len(kk),-1)
        del net,ema;torch.cuda.empty_cache()
        print(f'    {mode} fold{fi} done',flush=True)
    return OUT,DNR

if __name__=='__main__':
    set_det(1)
    subs=np.array([u for u in np.unique(S) if u not in EXCL])
    pmx=np.random.RandomState(0).permutation(len(subs))
    folds=[subs[pmx[i::NFOLD]] for i in range(NFOLD)]
    EV=np.concatenate([np.flatnonzero(S==u)[::7] for u in subs])
    SRC={'raw':V1.XIQ0,'chunk':BA.cache('chunk'),'window':BA.cache('window'),
         'envdphi':envdphi_cache()}
    print(f'  seed {SEED}  EP {EP}  arms {ARMS}   {len(subs)} subjects',flush=True)
    for a in ARMS:
        s_,_,r_,ro=CFG[a];print(f'    {a:9s} cand={s_:5s} rot={r_} roll={ro}')
    R={}
    for arm in ARMS:
        R[arm]=train_and_eval(arm,SRC,folds,EV)
    SE=S[EV];gtr=np.array([T.bpm(zn(G[i])) for i in EV])
    BEL=np.stack([zn(G[i]) for i in EV])
    print(f'\n  {"arm":7s} {"corr":>8s} {"donor":>8s} {"real":>8s} {"F1":>8s} '
          f'{"tmpl RMS":>9s} {"startR":>7s} {"rate sd":>8s} {"r(belt)":>9s}')
    per={}
    for arm in ARMS:
        OUT,DNR=R[arm]
        REC=np.stack([zn(OUT[i]) for i in EV]);DN=np.stack([zn(DNR[i]) for i in EV])
        cc=np.array([np.corrcoef(REC[j],BEL[j])[0,1] for j in range(len(EV))])
        dn=np.array([np.corrcoef(DN[j],BEL[j])[0,1] for j in range(len(EV))])
        f1=np.array([score(REC[j],BEL[j],FS)['breath_f1'] for j in range(len(EV))])
        pr=np.array([T.bpm(REC[j]) for j in range(len(EV))])
        tm=REC.mean(0);rms=float(np.sqrt((tm**2).mean()))
        ph=np.angle(hilbert(REC,axis=1))[:,:int(FS)].mean(1)
        sR=float(np.abs(np.mean(np.exp(1j*ph))))
        pc=lambda v:np.array([v[SE==u].mean() for u in subs])
        per[arm]=dict(cc=pc(cc),dn=pc(dn),f1=pc(f1))
        print(f'  {arm:7s} {pc(cc).mean():+8.4f} {pc(dn).mean():+8.4f} '
              f'{(pc(cc)-pc(dn)).mean():+8.4f} {pc(f1).mean():8.4f} '
              f'{rms:9.3f} {sR:7.3f} {pr.std():8.2f} {pearsonr(pr,gtr)[0]:+9.3f}')
    for b in ARMS[1:]:
        a=ARMS[0]
        d=per[b]['cc']-per[a]['cc'];dr=(per[b]['cc']-per[b]['dn'])-(per[a]['cc']-per[a]['dn'])
        print(f'\n  {b} minus {a}: corr {d.mean():+.4f} p={ttest_rel(per[b]["cc"],per[a]["cc"])[1]:.4f} '
              f'({int((d>0).sum())}/{len(subs)})   real gain {dr.mean():+.4f}   '
              f'F1 {(per[b]["f1"]-per[a]["f1"]).mean():+.4f} '
              f'p={ttest_rel(per[b]["f1"],per[a]["f1"])[1]:.4f}')
    tag='_'+'_'.join(ARMS) if len(ARMS)<=2 else ''
    np.savez_compressed(f'{PRE}/_rollaug{GTAG}{tag}_s{SEED}.npz',seed=SEED,arms=np.array(ARMS,dtype=object),
                        subs=subs,eval_idx=EV,
                        **{f'wv_{a}':R[a][0][EV] for a in ARMS},
                        **{f'dw_{a}':R[a][1][EV] for a in ARMS})
    print(f'\n  saved {PRE}/_rollaug{GTAG}{tag}_s{SEED}.npz')
