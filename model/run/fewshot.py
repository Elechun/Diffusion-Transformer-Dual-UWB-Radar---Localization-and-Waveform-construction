"""Few-shot adaptation to the test subject, calibrated at the head of the session and at the tail.

The generator takes event timing from the radar and fills everything else from the prior. The open
question is whether a small piece of the subject's OWN belt - a calibration block - moves the prior
toward that person. Head calibration is the deployable story: the first 84 s of a session are labelled,
the rest is not. Tail calibration is the same amount of the same subject's data taken from the other end;
it cannot be deployed (it adapts on the future), but if it works equally well the transfer is a property
of the person rather than of the neighbouring minutes.

Four arms, all built on the identical trained fold model (BR chunk + aug, S2 - the current baseline):

    none    the fold model as trained, untouched
    head    fine-tuned on the subject's first 8 windows (84 s of signal), scored on windows that do not
            overlap that block
    tail    the same with the last 8 windows
    other   fine-tuned on a DIFFERENT test subject's head block. Same extra gradient steps, same amount
            of belt, none of it this subject's - whatever 'head' gains over this is the person

Windows overlap (6 s hop, 42 s length), so an eval window is dropped whenever it shares any samples with
the calibration block - checked by explicit span intersection, not by counting. Comparisons are paired on
identical eval windows: head, other and none share one eval set; tail and none share another.

PROBE=1 verifies the bookkeeping and exits: no calibration/eval overlap anywhere, partner is never the
subject itself and never a training subject, and the eval sets agree across the arms being compared.
"""
import _path  # noqa: F401  (see _path.py)
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from scipy.stats import ttest_rel
from metrics import score,zn
import diffusion_model as D
import network as V1
import loss as LS
import br_candidates as BA
from determinism import set_det,gen_fixed
PRE=V1.PRE;GTAG=V1.GTAG;dev=D.dev
NC,CH,FS,W=D.NC,D.CH,D.FS,V1.W
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N
EXCL=tuple(int(v) for v in os.environ.get('EXCL','2,10,13,21,22').split(','))
SEED=int(os.environ.get('SEED','0'));EP=int(os.environ.get('EP','60'));NFOLD=int(os.environ.get('NFOLD','5'))   # NFOLD=10: 10-fold subject CV (use a WTAG such as _k10 so the 5-fold weights are not overwritten)
KCAL=int(os.environ.get('KCAL','8'))          # calibration windows; 8 rows = 84 s of signal
ASTEP=int(os.environ.get('ASTEP','100'))      # adaptation gradient steps
ALR=float(os.environ.get('ALR','5e-5'))
PROBE=os.environ.get('PROBE','0')=='1'
ARMS=['none','head','tail','other']

HOP_S=getattr(D,'HOP_S',6.0);STRIDE=int(round(42/HOP_S))
def spans(rows_local):
    """each local row i covers [hop*i, hop*i+42] seconds of its subject's record"""
    return [(HOP_S*i,HOP_S*i+42) for i in rows_local]
def overlaps(a,b): return a[0]<b[1] and b[0]<a[1]

def plan(subs):
    """per subject: global calibration rows (head/tail), partner, and the eval rows for each arm"""
    P={}
    folds=[subs[np.random.RandomState(0).permutation(len(subs))[i::NFOLD]] for i in range(NFOLD)]
    for fi,te in enumerate(folds):
        for k,u in enumerate(te):
            rows=np.flatnonzero(S==u);L=len(rows)
            loc_ev=np.arange(L)[::STRIDE]
            head_loc=np.arange(0,KCAL);tail_loc=np.arange(L-KCAL,L)
            hs=spans(head_loc);ts=spans(tail_loc)
            ev_head=[i for i in loc_ev if not any(overlaps(spans([i])[0],s) for s in hs)]
            ev_tail=[i for i in loc_ev if not any(overlaps(spans([i])[0],s) for s in ts)]
            partner=te[(k+1)%len(te)]                     # another held-out subject, never trained on
            P[int(u)]=dict(fold=fi,rows=rows,
                           cal_head=rows[head_loc],cal_tail=rows[tail_loc],
                           ev_head=rows[ev_head],ev_tail=rows[ev_tail],ev_all=rows[loc_ev],
                           partner=int(partner))
    return P,folds

def probe(P,subs):
    bad=0
    for u,d in P.items():
        rows=d['rows'];base=rows[0]
        for cal,ev in (('cal_head','ev_head'),('cal_tail','ev_tail')):
            cs=spans(d[cal]-base)
            for e in d[ev]-base:
                es=spans([e])[0]
                if any(overlaps(es,c) for c in cs):
                    print(f'  OVERLAP sub{u:02d} {cal} row {e}');bad+=1
        assert d['partner']!=u
        assert S[P[d['partner']]['rows'][0]]==d['partner']
        assert P[d['partner']]['fold']==d['fold'],'partner trained in this fold'
        assert set(d['ev_head'])<=set(d['ev_all']) and set(d['ev_tail'])<=set(d['ev_all'])
    ne_h=[len(P[u]['ev_head']) for u in P];ne_t=[len(P[u]['ev_tail']) for u in P]
    print(f'  probe: {len(P)} subjects, overlaps={bad}')
    print(f'  eval windows kept per subject: head {min(ne_h)}..{max(ne_h)}  tail {min(ne_t)}..{max(ne_t)}')
    print(f'  calibration = {KCAL} rows = {HOP_S*(KCAL-1)+42:g} s of signal at each end (hop {HOP_S:g} s)')
    assert bad==0

ADAPT_PARAMS=os.environ.get('ADAPT_PARAMS','all')   # all | film (AdaLN modulation + output head only: amplitude/morphology,
                                                    # no re-timing) | out (output head only) | none
def _adapt_subset(net):
    if ADAPT_PARAMS=='all': return list(net.parameters())
    keep=[]
    for n_,p_ in net.named_parameters():
        if ADAPT_PARAMS=='out' and ('out.' in n_ or 'nout' in n_): keep.append(p_)
        elif ADAPT_PARAMS=='film' and ('.ada.' in n_ or 'out.' in n_ or 'nout' in n_): keep.append(p_)
    return keep
def adapt(ema,rows_cal,XIQ,Cn,Pn,Gt2,tag,cu=None):
    """a copy of the fold model, fine-tuned on the calibration rows only.
    cu: optional (1,2) z-scored calibration-statistics vector (CALSTATS=1), fed to every step like at sampling time"""
    net=copy.deepcopy(ema)
    if ADAPT_PARAMS=='none':
        net.eval();[p.requires_grad_(False) for p in net.parameters()];return net
    [p.requires_grad_(False) for p in net.parameters()];sub=_adapt_subset(net);[p.requires_grad_(True) for p in sub]
    opt=torch.optim.AdamW(sub,ALR,weight_decay=1e-4)
    xt=torch.tensor(XIQ[rows_cal],device=dev);ct=torch.tensor(Cn[rows_cal],device=dev)
    pt=torch.tensor(Pn[rows_cal],device=dev);zt=torch.tensor(ZONE[rows_cal],device=dev)
    gt=torch.tensor(Gt2[rows_cal],device=dev)
    import zlib;torch.manual_seed(zlib.crc32(tag.encode())%2**31)   # stable across processes, unlike hash()
    net.train()
    for _ in range(ASTEP):
        B=len(rows_cal)
        xin=V1.rotate(xt) if os.environ.get('SRC','chunk')!='envdphi' else xt
        tok,_,lz=net.cond(xin,ct,pt)
        t=torch.randint(0,D.T_DIFF,(B,),device=dev);eps=torch.randn_like(gt)
        a=D.ab[t][:,None,None];xn=a.sqrt()*gt+(1-a).sqrt()*eps
        e=net(xn,t,tok,cu=(None if cu is None else cu.expand(B,-1)));l=Fn.mse_loss(e,eps)
        x0=(xn-(1-a).sqrt()*e)/a.sqrt()
        l=l+Fn.cross_entropy(lz.reshape(-1,14),zt.reshape(-1),ignore_index=-1)
        l=l+LS.loss_S2(x0,gt,B)
        opt.zero_grad();l.backward();nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad],1.);opt.step()
    net.eval();[p.requires_grad_(False) for p in net.parameters()]
    return net

@torch.no_grad()
def gen(net,rows,XIQ,Cn,Pn):
    out=np.zeros((len(rows),W),np.float32)
    for b in range(0,len(rows),16):
        kk=rows[b:b+16]
        ci=torch.tensor(Cn[kk],device=dev);pi=torch.tensor(Pn[kk],device=dev)
        xi=torch.tensor(XIQ[kk],device=dev)
        tok,_,_=net.cond(xi,ci,pi);tku,_,_=net.cond(torch.zeros_like(xi),ci,pi)
        out[b:b+16]=gen_fixed(net,tok,tku,kk,1).cpu().numpy().reshape(len(kk),-1)
    return out

if __name__=='__main__':
    set_det(1)
    subs=np.array([u for u in np.unique(S) if u not in EXCL])
    P,folds=plan(subs)
    probe(P,subs)
    if PROBE: raise SystemExit('  probe only, exiting')
    SRC=os.environ.get('SRC','chunk')
    if SRC=='envdphi':
        import augment as RAmod
        XIQ=RAmod.envdphi_cache()          # rotation-invariant winner; rotate() must not touch it
    else:
        XIQ=BA.cache(SRC)
    Gt2=G.reshape(N,NC,CH).astype(np.float32)
    print(f'  seed {SEED}  EP {EP}  adapt {ASTEP} steps @ lr {ALR}  cal {KCAL} rows',flush=True)
    WV={a:np.zeros((N,W),np.float32) for a in ARMS}
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
                xin=V1.rotate(xt[j]) if os.environ.get('SRC','chunk')!='envdphi' else xt[j]
                tok,_,lz=net.cond(xin,ct[j],pt[j])
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
            d=P[int(u)]
            WV['none'][d['ev_all']]=gen(ema,d['ev_all'],XIQ,Cn,Pn)
            nh=adapt(ema,d['cal_head'],XIQ,Cn,Pn,Gt2,f'h{u}s{SEED}')
            WV['head'][d['ev_head']]=gen(nh,d['ev_head'],XIQ,Cn,Pn);del nh
            nt=adapt(ema,d['cal_tail'],XIQ,Cn,Pn,Gt2,f't{u}s{SEED}')
            WV['tail'][d['ev_tail']]=gen(nt,d['ev_tail'],XIQ,Cn,Pn);del nt
            no=adapt(ema,P[d['partner']]['cal_head'],XIQ,Cn,Pn,Gt2,f'o{u}s{SEED}')
            WV['other'][d['ev_head']]=gen(no,d['ev_head'],XIQ,Cn,Pn);del no
            torch.cuda.empty_cache()
        del net,ema;torch.cuda.empty_cache();print(f'  fold{fi} done',flush=True)

    sg=lambda a,b: float(np.corrcoef(zn(a),zn(b))[0,1])
    res={}
    for arm,ev in (('head','ev_head'),('tail','ev_tail'),('other','ev_head')):
        da=[];dn=[];fa=[];fn=[]
        for u in subs:
            d=P[int(u)];rows=d[ev]
            da.append(np.mean([sg(WV[arm][r],G[r]) for r in rows]))
            dn.append(np.mean([sg(WV['none'][r],G[r]) for r in rows]))
            fa.append(np.mean([score(WV[arm][r],G[r],FS)['breath_f1'] for r in rows]))
            fn.append(np.mean([score(WV['none'][r],G[r],FS)['breath_f1'] for r in rows]))
        res[arm]=dict(cc=np.array(da),cc0=np.array(dn),f1=np.array(fa),f10=np.array(fn))
        dd=res[arm]['cc']-res[arm]['cc0'];df=res[arm]['f1']-res[arm]['f10']
        print('  %-6s corr %+0.4f -> %+0.4f  d %+0.4f p=%.4f (%d/%d)   F1 d %+0.4f p=%.4f'%(
            arm,res[arm]['cc0'].mean(),res[arm]['cc'].mean(),dd.mean(),ttest_rel(res[arm]['cc'],res[arm]['cc0'])[1],
            int((dd>0).sum()),len(subs),df.mean(),ttest_rel(res[arm]['f1'],res[arm]['f10'])[1]))
    hh=res['head']['cc']-res['other']['cc']
    print('  head minus other (the subject-specific part): %+0.4f  p=%.4f  (%d/%d)'%(
        hh.mean(),ttest_rel(res['head']['cc'],res['other']['cc'])[1],int((hh>0).sum()),len(subs)))
    np.savez_compressed(f'{PRE}/_fewshot_ht{GTAG}_{os.environ.get("SRC","chunk")}_s{SEED}.npz',seed=SEED,subs=subs,
                        kcal=KCAL,astep=ASTEP,alr=ALR,excl=np.array(EXCL),
                        **{f'wv_{a}':WV[a] for a in ARMS},
                        **{f'{a}_{k}':res[a][k] for a in res for k in res[a]})
    print(f'  saved {PRE}/_fewshot_ht{GTAG}_{os.environ.get("SRC","chunk")}_s{SEED}.npz')
