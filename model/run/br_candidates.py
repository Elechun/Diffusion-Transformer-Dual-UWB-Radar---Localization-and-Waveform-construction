"""Feed the model BR-compensated candidates instead of raw ones, and see whether the waveform improves.

Backward Reasoning is a PREPROCESSING step in the paper: the four extracted signals are phase-compensated
before the model ever sees them, to undo the initial phase from the user's distance and the posture-induced
phase error. What was measured here first is what it can and cannot do:

    it CAN find the axis         the I-Q cloud goes from squashed to spread, exactly as the paper's Fig 7
    it CANNOT find the direction  Re(z e^-i(theta+pi)) = -Re(z e^-i theta), so argmax|V| is exactly
                                  indifferent between theta and theta+180. Measured: 0.000% amplitude
                                  difference, and it lands on the belt-matching sign 50.9% of the time

That sign ambiguity is NOT introduced by BR. Neighbouring chunks disagree on the belt-matching sign 48.3%
of the time with per-chunk BR, 46.5% with per-window BR, and 46.0% in the current pipeline which applies no
BR at all. The sign flips because the body turns while walking; BR neither causes nor fixes it.

So the open question is whether fixing the AXIS helps, given the sign stays random either way. The current
pipeline answers the same problem differently - it keeps I and Q and rotates by a random angle during
training, forcing the model to be invariant to the angle rather than fixing it. Those are alternatives, and
they can be combined:

    raw + aug        what runs today: candidates as extracted, random rotation each step
    BR chunk + aug   BR axis per 3 s chunk, then still augmented
    BR chunk, no aug BR axis per 3 s chunk, angle left fixed - the paper's intent
    BR window + aug  one BR axis per 42 s window per candidate

Everything else is held: same seed, same folds, same architecture, the base loss, the same row-tied start
noise shared across arms. Scored with a SIGNED correlation - an absolute value was scoring inversions as
successes, and 31-37% of pieces are inverted, so it mattered.
"""
import _path  # noqa: F401  (see _path.py)
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from scipy.stats import ttest_rel
from normalize import zn
from metrics import score
import diffusion_model as D
import network as V1
from determinism import set_det,gen_fixed
import loss as LS
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
EP=int(os.environ.get('EP','60'));SEED=int(os.environ.get('SEED','0'));NFOLD=5
KD=int(os.environ.get('KD','4'));NTH=int(os.environ.get('NTH','180'))
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N;NCAND=V1.XIQ0.shape[1]//2
# One place for the cohort. sub22's BIOPAC amplifier sat on its rail for 28.25% of the record - the median
# subject is at 0.01% - so its ground truth is clipped, not merely noisy. Excluded on that basis, which is
# a property of the recording and not of anything this model produced.
EXCL=tuple(int(v) for v in os.environ.get('EXCL','2,10,13,21,22').split(','))
TAG=os.environ.get('TAG','')
# RESYNC=1 replaces the belt of every hand-aligned subject with the piece the operator cut by eye.
# The offsets live in preprocessed/matlab_sync/subXX_manual_sync.npz; earlier campaigns are kept in a
# prev_campaign/ subfolder and are deliberately NOT picked up by the glob.
# 0 = the pipeline's own belts, 1 = the hand-aligned ones, 2 = the same subjects re-cut at a
# RANDOM offset from the same allowed range, which is the control the hand alignment needs.
RESYNC=int(os.environ.get('RESYNC','0'))
def manual_belts():
    """G with the hand-aligned belts substituted, plus the subjects and rows that actually changed.

    Row i of a subject starts at i*HOP - verified on all 14 at corr 1.0000, so a window is addressed by
    its rank. A window whose 42 s runs past the end of the hand-cut belt keeps its original ground truth
    rather than being padded, which is what happens when the belt recording stopped before the radar did.
    """
    import glob
    G2=G.copy().astype(np.float32);hit=[];HOP=CH*2;did={}
    for f in sorted(glob.glob(f'{PRE}/matlab_sync/sub*_manual_sync.npz')):
        z=np.load(f,allow_pickle=True);u=int(z['sub'])
        if u in EXCL: continue
        man=np.asarray(z['gt_manual'],float);rows=np.flatnonzero(S==u)
        if not len(rows): continue
        ok=0
        for i,r in enumerate(rows):
            s0=i*HOP
            if s0+W>len(man): continue
            seg=man[s0:s0+W]
            if seg.std()<1e-9: continue
            G2[r]=((seg-seg.mean())/(seg.std()+1e-9)).astype(np.float32);hit.append(int(r));ok+=1
        did[u]=(ok,len(rows),float(z['offset_seconds']))
    return G2,np.array(sorted(hit)),did
def br_compensate(X,per):
    """X (N,8,NC,CH) -> the same with each candidate rotated onto its own max-amplitude axis.

    per='chunk' solves one angle per 3 s chunk, per='window' one angle for the whole 42 s. Both keep I and
    Q; nothing is projected away, so no information is discarded - only the frame is changed.
    """
    Y=np.empty_like(X);TH=np.arange(NTH)*np.pi/NTH
    for c in range(NCAND):
        z=X[:,2*c].astype(np.float64)+1j*X[:,2*c+1].astype(np.float64)     # (N,NC,CH)
        if per=='window':
            zf=z.reshape(len(X),-1)
            P=np.real(zf[None]*np.exp(-1j*TH[:,None,None]))                # (A,N,W)
            a=np.argmax(P.max(-1)-P.min(-1),0)                             # (N,)
            zr=(zf*np.exp(-1j*TH[a])[:,None]).reshape(z.shape)
        else:
            P=np.real(z[None]*np.exp(-1j*TH[:,None,None,None]))            # (A,N,NC,CH)
            a=np.argmax(P.max(-1)-P.min(-1),0)                             # (N,NC)
            zr=z*np.exp(-1j*TH[a])[...,None]
        Y[:,2*c]=zr.real.astype(np.float32);Y[:,2*c+1]=zr.imag.astype(np.float32)
    return Y
def cache(per):
    f=f'{PRE}/_iq_candidates{GTAG}_br{per}.npy'
    if not os.path.exists(f):
        print(f'  building {os.path.basename(f)} ...',flush=True)
        np.save(f,br_compensate(V1.XIQ0,per))
    return np.load(f).astype(np.float32)
# A 2x2: the candidate frame crossed with the loss shape, so both main effects and their interaction come
# out of one run. The two arms dropped from the first pass are settled - "BR without augmentation" was
# -0.0522 (p=0.0013), worse than everything by a wide margin, and "BR per window" landed on top of "BR per
# chunk" (-0.0022 vs -0.0014). Neither needs three seeds spent on it.
ARMS=[('raw + aug, base','raw',True,LS.loss_base),
      ('BR chunk + aug, base','chunk',True,LS.loss_base),
      ('raw + aug, S2','raw',True,LS.loss_S2),
      ('BR chunk + aug, S2','chunk',True,LS.loss_S2)]
PICK=os.environ.get('ARMS')
if PICK: ARMS=[a for a in ARMS if a[0] in PICK.split('|')]
def run_arm(name,src,aug,lossf,XIQ):
    subs=np.array([u for u in np.unique(S) if u not in EXCL])
    pmx=np.random.RandomState(0).permutation(len(subs));folds=[subs[pmx[i::NFOLD]] for i in range(NFOLD)]
    EV=np.concatenate([np.flatnonzero(S==u)[::7] for u in subs])
    rs=np.random.RandomState(1);out=np.isin(S,EXCL)
    DON=np.array([rs.choice(np.flatnonzero((S!=S[i])&~out)) for i in range(N)])
    Gt2=GG.reshape(N,NC,CH).astype(np.float32)
    OUT=np.zeros((N,KD,W),np.float32);DNR=np.zeros((N,W),np.float32)
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
                xin=V1.rotate(xt[j]) if aug else xt[j]
                tok,_,lz=net.cond(xin,ct[j],pt[j])
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
                    OUT[kk,d]=gen_fixed(ema,tok,tku,kk,1,seed=7+d).cpu().numpy().reshape(len(kk),-1)
                xd=torch.tensor(XIQ[DON[kk]],device=dev)
                t2,_,_=ema.cond(xd,ci,pi);u2,_,_=ema.cond(torch.zeros_like(xd),ci,pi)
                DNR[kk]=gen_fixed(ema,t2,u2,kk,1).cpu().numpy().reshape(len(kk),-1)
        del net,ema;torch.cuda.empty_cache();print(f'    {name}  fold{fi} done',flush=True)
    SE=S[EV];Y=OUT[EV]
    sg=lambda a,b: float(np.corrcoef(zn(a),zn(b))[0,1])
    per=lambda v:np.array([np.asarray(v)[SE==u].mean() for u in subs])
    cc=np.array([sg(Y[j,0],GG[i]) for j,i in enumerate(EV)])
    mn=np.array([zn(Y[j]).mean(0) for j in range(len(EV))])
    ccm=np.array([sg(mn[j],GG[i]) for j,i in enumerate(EV)])
    dn=np.array([sg(DNR[i],GG[i]) for i in EV])
    f1=np.array([score(Y[j,0],GG[i],FS)['breath_f1'] for j,i in enumerate(EV)])
    df=np.array([score(DNR[i],GG[i],FS)['breath_f1'] for i in EV])
    return dict(subs=subs,eval_idx=EV,cc=per(cc),ccm=per(ccm),dn=per(dn),f1=per(f1),df=per(df),
                inv=100*np.mean(cc<0),wv=Y[:,0],wvm=mn,dw=DNR[EV])
if __name__=='__main__':
    set_det(1)
    GG=G.astype(np.float32)
    if RESYNC:
        if RESYNC==2:
            import gt_audit as GU
            GG,hit,did=GU.decoy_belts(EXCL,seed=SEED)
            print(f'  RESYNC=2 DECOY: {len(hit)} windows across {len(did)} subjects re-cut at a RANDOM '
                  f'offset (seed {SEED})')
        else:
            GG,hit,did=manual_belts()
            print(f'  RESYNC on: {len(hit)} windows across {len(did)} subjects use a hand-aligned belt')
        for u in sorted(did):
            ok,tot,off=did[u]
            print(f'    sub{u:02d}  {off:+6.2f} s   {ok}/{tot} windows replaced')
    else:
        print('  RESYNC off: every belt is the pipeline\'s own alignment')
    print(f'  cohort {len(np.unique(S))-len([u for u in np.unique(S) if u in EXCL])} subjects, '
          f'excluding {EXCL}')
    print(f'  seed {SEED}  EP {EP}  {KD} draws   scored with a SIGNED correlation')
    print(f'  arms: '+' | '.join(n for n,_,_,_ in ARMS),flush=True)
    SRC={'raw':V1.XIQ0}
    for _,s_,_,_ in ARMS:
        if s_ not in SRC: SRC[s_]=cache(s_)
    R={}
    for name,s_,aug,lf in ARMS:
        print(f'  --- {name} ---',flush=True);R[name]=run_arm(name,s_,aug,lf,SRC[s_])
    print(f'\n  {"arm":20s} {"signed":>8s} {"donor":>8s} {"real":>8s} {"F1":>8s} {"real":>8s} '
          f'{"inverted":>9s} {"K-mean":>8s}')
    for name,_,_,_ in ARMS:
        r=R[name]
        print(f'  {name:20s} {r["cc"].mean():+8.4f} {r["dn"].mean():+8.4f} '
              f'{r["cc"].mean()-r["dn"].mean():+8.4f} {r["f1"].mean():8.4f} '
              f'{r["f1"].mean()-r["df"].mean():+8.4f} {r["inv"]:8.1f}% {r["ccm"].mean():+8.4f}')
    b=ARMS[0][0]
    print(f'\n  paired against "{b}" over {len(R[b]["subs"])} subjects')
    for name,_,_,_ in ARMS:
        if name==b: continue
        x,y=R[name]['cc'],R[b]['cc'];rx=R[name]['cc']-R[name]['dn'];ry=R[b]['cc']-R[b]['dn']
        print(f'  {name:20s} signed {x.mean()-y.mean():+.4f} p {ttest_rel(x,y).pvalue:.4f} '
              f'({int((x>y).sum())}/{len(x)})   real {rx.mean()-ry.mean():+.4f} p {ttest_rel(rx,ry).pvalue:.4f}')
    np.savez_compressed(f'{PRE}/_br_arms{GTAG}{TAG}_s{SEED}.npz',seed=SEED,kd=KD,
                        excl=np.array(EXCL),resync=int(RESYNC),
                        arms=np.array([n for n,_,_,_ in ARMS],dtype=object),
                        subs=R[ARMS[0][0]]['subs'],eval_idx=R[ARMS[0][0]]['eval_idx'],
                        **{f'{k}_{i}':R[n][k] for i,(n,_,_,_) in enumerate(ARMS)
                           for k in ('cc','ccm','dn','f1','df','wv','wvm','dw')})
    print(f'\n  saved {PRE}/_br_arms{GTAG}{TAG}_s{SEED}.npz\n')
