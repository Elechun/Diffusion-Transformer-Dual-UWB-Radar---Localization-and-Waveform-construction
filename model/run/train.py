"""Train the envdphi_s2 fold models once and keep the EMA weights on disk.

Every sampling-time experiment so far (guidance strength, sigma, calibration size, bridges) has re-run
the identical 5-fold training just to have a model to sample from - forty minutes of GPU to answer a
five-minute question. This trains with the exact seeds and loss the specguide/assembly runs used and
saves each fold's EMA state_dict, so sweeps become sampling-only.

    preprocessed/foldw/_envdphi_s2_s{SEED}_f{fold}.pt

fold_norm and the fold membership are deterministic (RandomState(0)), so a loader only needs the seed.
"""
import _path  # noqa: F401  (see _path.py)
import os,re
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
import diffusion_model as D
import network as V1
import loss as LS
import augment as RA
from determinism import set_det
PRE=V1.PRE;dev=D.dev
NC,CH=D.NC,D.CH
S,G,ZONE=D.S,D.G,D.ZONE;N=V1.N
EXCL=tuple(int(v) for v in os.environ.get('EXCL','2,10,13,21,22').split(','))
SEED=int(os.environ.get('SEED','0'));EP=int(os.environ.get('EP','60'));NFOLD=int(os.environ.get('NFOLD','5'))   # NFOLD=10: 10-fold subject CV (use a WTAG such as _k10 so the 5-fold weights are not overwritten)
OUTD=f'{PRE}/foldw';os.makedirs(OUTD,exist_ok=True)
# RADAR=both (default) | com | tv  - single-radar ablation: the other radar's four channels are zeroed
# for the whole training, so the model only ever sees one radar (architecture unchanged)
RADAR=os.environ.get('RADAR','both')
RTAG='' if RADAR=='both' else f'_{RADAR}'
# RANDCROP=1: random-crop training on a fine-hop dataset (e.g. hop 1 s). Each epoch draws, uniformly
# without replacement, as many training rows as a 6 s grid would hold (n * hop / 6), so the number of
# optimiser steps matches the reference while the same 42 s of signal is cut at a different breathing
# phase every epoch - the phase shortcut of a fixed grid is not available.
RANDCROP=os.environ.get('RANDCROP','0')=='1'
if RANDCROP: RTAG+='_rc'
# ROT=1: random global I/Q rotation per row per epoch (network.rotate) - the augmentation MoRe-Fi /
# IQ-VED use to absorb the arbitrary phase reference. Only meaningful with a raw I/Q input cache
# (INCACHE=iq_candidates); the deployed envdphi channels are rotation-invariant by construction, so rotating them
# is a no-op on |z| and a sign change on dphi. The input cache is tagged too, so an I/Q run cannot overwrite the
# envdphi weights of the same seed.
ROT=os.environ.get('ROT','0')=='1'
if os.environ.get('INCACHE',''): RTAG+=f"_in{os.environ['INCACHE']}"
if ROT: RTAG+='_rot'
RTAG+=os.environ.get('WTAG','')
assert os.environ.get('NFOLD','5')=='5' or '_k'+os.environ['NFOLD'] in RTAG, 'NFOLD!=5 needs WTAG containing _k<NFOLD> (never overwrite 5-fold weights)'
EMA_DECAY=float(os.environ.get('EMA_DECAY','0.999'))
if EMA_DECAY!=0.999: RTAG+=f'_ema{EMA_DECAY:g}'
# PEAKLAM>0: add the belt-peak / model-peak distance loss on x0 (analytic phase of the band-limited x0 must be 0 at belt
# peak samples and pi at belt valley samples; cost 1-cos(2*pi*dt/period)). Tag _pk{PEAKLAM}. Same term as
# an internal analysis script LOSS=peak.
ALIGNLAM=float(os.environ.get('ALIGNLAM','0'))   # RF-Carer L1: latent features of ADJACENT range bins must agree (respiration is common to adjacent bins, noise is not); pairs = channels (c, c+2) inside each radar block of the multibin cache
PEAKLAM=float(os.environ.get('PEAKLAM','0'))
PEAKLOSS=os.environ.get('PEAKLOSS','phase')      # phase: analytic phase at belt events | adj: 1:1 peak ADJUSTMENT (soft peak mass per belt breath) | lagdup: adj + free per-window lag + asymmetric peak-count penalty
if PEAKLAM>0: RTAG+=({'phase':f'_pk{PEAKLAM:g}','adj':f'_pa{PEAKLAM:g}'}.get(PEAKLOSS,f'_pl{PEAKLAM:g}w{float(os.environ.get("PKW","24")):g}g{float(os.environ.get("PKLAG","1.0")):g}d{float(os.environ.get("PKDUP","1.0")):g}'))
if ALIGNLAM>0: RTAG+=f'_al{ALIGNLAM:g}'
_fr=torch.fft.rfftfreq(D.NC*D.CH,1/D.FS);BANDR=((_fr>=0.1)&(_fr<=0.6)).float().to(dev)
def peak_adjust_loss(p,pk,vl,per,tau=0.3):
    """see an internal analysis script.peak_adjust_loss: expected squared distance (in periods, x4) of the model's soft peak mass
    inside each belt breath (between two belt valleys) to that breath's belt peak; one model peak per belt peak by construction"""
    B,n=p.shape;t=torch.arange(n,device=p.device,dtype=p.dtype)
    z=torch.fft.irfft(torch.fft.rfft(p,dim=-1)*BANDR,n=n,dim=-1);z=(z-z.mean(-1,keepdim=True))/(z.std(-1,keepdim=True)+1e-6)
    lo=vl[:,:-1];hi=vl[:,1:];valid=(pk>=0)&(hi>lo)
    m=(t[None,None,:]>=lo[:,:,None])&(t[None,None,:]<hi[:,:,None])
    w=torch.softmax((z[:,None,:]/tau).masked_fill(~m,-1e4),-1)
    d2=((t[None,None,:]-pk[:,:,None])/per[:,None,None])**2
    return (((w*d2).sum(-1)*4.0)*valid).sum()/(valid.sum()+1e-6)
PKW=float(os.environ.get('PKW','24'))            # lagdup: length of the short scoring window, in seconds
PKLAG=float(os.environ.get('PKLAG','1.0'))       # lagdup: the window may slide by up to +-PKLAG s before it is scored (sync)
PKDUP=float(os.environ.get('PKDUP','1.0'))       # lagdup: weight of the peak-count penalty; over-counting is charged twice
def _smooth_peak_count(z,lo,hi,thr=0.0,tau=0.005):
    """differentiable count of local maxima of z above thr inside each [lo,hi) interval"""
    B,n=z.shape;t=torch.arange(n,device=z.device,dtype=z.dtype)
    zp=Fn.pad(z[:,None,:],(1,1),mode='replicate')[:,0]
    up=torch.sigmoid((z-zp[:,:-2])/tau);dn=torch.sigmoid((z-zp[:,2:])/tau);hg=torch.sigmoid((z-thr)/tau)
    q=up*dn*hg                                     # (B,n) soft indicator of "this sample is a peak"
    m=((t[None,None,:]>=lo[:,:,None])&(t[None,None,:]<hi[:,:,None])).to(z.dtype)
    return (q[:,None,:]*m).sum(-1)                 # (B,K)
def peak_lagdup_loss(p,pk,vl,per,win_s=None,lag_s=None,dupw=None,tau=0.3):
    """short-window peak-distance loss with a free per-window lag (sync) plus an asymmetric peak-count penalty.
    Inside each belt breath (two consecutive belt valleys) the model should put exactly ONE peak, as close as possible to
    the belt peak; a window may first slide by up to +-lag_s to absorb a constant offset, and a breath that receives two
    model peaks is charged twice as much as one that receives none."""
    win_s=PKW if win_s is None else win_s;lag_s=PKLAG if lag_s is None else lag_s;dupw=PKDUP if dupw is None else dupw
    B,n=p.shape;t=torch.arange(n,device=p.device,dtype=p.dtype)
    z=torch.fft.irfft(torch.fft.rfft(p,dim=-1)*BANDR,n=n,dim=-1);z=(z-z.mean(-1,keepdim=True))/(z.std(-1,keepdim=True)+1e-6)
    lo=vl[:,:-1];hi=vl[:,1:];valid=((pk>=0)&(hi>lo)).float()
    m=(t[None,None,:]>=lo[:,:,None])&(t[None,None,:]<hi[:,:,None])
    w=torch.softmax((z[:,None,:]/tau).masked_fill(~m,-1e4),-1)          # (B,K,n) soft peak mass per belt breath
    Lw=max(1,int(round(win_s*D.FS)));nw=max(1,int(np.ceil(n/Lw)))
    widx=torch.clamp((pk/Lw).long(),0,nw-1)                              # which short window each belt breath falls in
    lags=torch.arange(-round(lag_s*D.FS),round(lag_s*D.FS)+1,max(1,int(round(0.25*D.FS))),device=p.device,dtype=p.dtype)
    dist=[]
    for dl in lags:
        d2=((t[None,None,:]-(pk+dl)[:,:,None])/per[:,None,None])**2
        dist.append(((w*d2).sum(-1)*4.0)*valid)                          # (B,K) cost of every breath at this lag
    dist=torch.stack(dist,-1)                                            # (B,K,nlag)
    oh=Fn.one_hot(widx,nw).to(p.dtype)*valid[:,:,None]                   # (B,K,nw)
    per_win=torch.einsum('bkl,bkw->bwl',dist,oh);cntw=oh.sum(1)          # (B,nw,nlag),(B,nw)
    best=per_win.min(-1).values                                          # each short window keeps its best lag
    lossd=(best.sum(1)/(cntw.sum(1)+1e-6)).mean()
    c=_smooth_peak_count(z,lo,hi)                                        # (B,K)
    over=torch.relu(c-1.0)**2;under=torch.relu(1.0-c)**2
    lossc=(((2.0*over+under)*valid).sum(1)/(valid.sum(1)+1e-6)).mean()
    return lossd+dupw*lossc
STATLAM=float(os.environ.get('STATLAM','0'))     # weight of the GT-STATISTICS loss (not the waveform itself)
STATSET=os.environ.get('STATSET','rate,count,acf')
STATWIN=float(os.environ.get('STATWIN','0'))     # >0: match the statistics on SLIDING sub-windows of this length (s)
STATHOP=float(os.environ.get('STATHOP','6'))     # stride of those sub-windows
if STATLAM>0: RTAG+=f'_st{STATLAM:g}'+(f'w{STATWIN:g}' if STATWIN>0 else '')+('' if STATSET=='rate,count,acf' else '_'+STATSET.replace(',',''))
def _soft_rate(v,lo=0.1,hi=0.6,tau=0.05):
    """differentiable breathing rate of a batch of windows: soft-argmax of the band-limited power spectrum, in bpm"""
    n=v.shape[-1];F=torch.fft.rfft(v-v.mean(-1,keepdim=True),dim=-1);P=(F.real**2+F.imag**2)
    f=torch.fft.rfftfreq(n,1/D.FS).to(v.device);m=(f>=lo)&(f<=hi)
    P=P[:,m];fb=f[m]
    w=torch.softmax(torch.log(P+1e-9)/tau,dim=-1)
    return (w*fb[None,:]).sum(-1)*60.0
def _acf_at(v,lag):
    """autocorrelation of each row at its own integer lag (the belt's own breathing period)"""
    v=v-v.mean(-1,keepdim=True);v=v/(v.std(-1,keepdim=True)+1e-6);n=v.shape[-1]
    out=[]
    for i in range(v.shape[0]):
        L=int(lag[i].item())
        L=max(2,min(L,n-2));out.append((v[i,:-L]*v[i,L:]).mean())
    return torch.stack(out)
def _stat_one(z,rate_gt,cnt_gt,per_gt):
    l=0.0
    if 'rate' in STATSET: l=l+((_soft_rate(z)-rate_gt)/6.0).pow(2).mean()
    if 'count' in STATSET:
        lo=torch.zeros(len(z),1,device=z.device);hi=torch.full((len(z),1),float(z.shape[-1]),device=z.device)
        c=_smooth_peak_count(z,lo,hi)[:,0];l=l+((c-cnt_gt)/4.0).pow(2).mean()
    if 'acf' in STATSET: l=l+(1.0-_acf_at(z,per_gt)).mean()
    return l
def stat_loss(p,rate_gt,cnt_gt,per_gt):
    """match the belt's SUMMARY statistics: its rate, how many breaths it contains, how periodic it is at that period.
    None of these says WHERE a breath is - only how many and how fast. With STATWIN>0 the statistics are matched on
    sliding sub-windows, so a rate that changes inside the window has to be followed rather than averaged away."""
    z=torch.fft.irfft(torch.fft.rfft(p,dim=-1)*BANDR,n=p.shape[-1],dim=-1)
    z=(z-z.mean(-1,keepdim=True))/(z.std(-1,keepdim=True)+1e-6)
    if STATWIN<=0: return _stat_one(z,rate_gt[:,0],cnt_gt[:,0],per_gt[:,0])
    L=int(round(STATWIN*D.FS));H=int(round(STATHOP*D.FS));n=z.shape[-1];l=0.0;k=0
    for q,s0 in enumerate(range(0,n-L+1,H)):
        if q>=rate_gt.shape[1]: break
        l=l+_stat_one(z[:,s0:s0+L],rate_gt[:,q],cnt_gt[:,q],per_gt[:,q]);k+=1
    return l/max(1,k)
def stat_tables():
    """per training row (and per sliding sub-window if STATWIN>0): the belt's rate, breath count and median period.
    Three numbers per sub-window, no timing."""
    W_=G.shape[1];L=W_ if STATWIN<=0 else int(round(STATWIN*D.FS));H=int(round(STATHOP*D.FS))
    starts=[0] if STATWIN<=0 else list(range(0,W_-L+1,H))
    R=np.zeros((N,len(starts)),np.float32);C=np.zeros((N,len(starts)),np.float32);P=np.full((N,len(starts)),3.0*D.FS,np.float32)
    for i in range(N):
        for q,s0 in enumerate(starts):
            seg=G[i][s0:s0+L];a=ev(seg,1);C[i,q]=len(a);R[i,q]=len(a)/(len(seg)/D.FS)*60
            if len(a)>2: P[i,q]=np.median(np.diff(a))
    return R,C,P
def event_tables(sign,KMAX=24):
    W_=D.NC*D.CH;PK=np.full((N,KMAX),-1,np.float32);VL=np.zeros((N,KMAX+1),np.float32);PER=np.ones(N,np.float32)*D.FS*3
    for i in range(N):
        a=ev(G[i],sign);b=ev(G[i],-sign)
        if len(a)<2: continue
        PER[i]=np.median(np.diff(a));edges=[0.0]+[float(x) for x in b if a[0]<x<a[-1]]+[float(W_)];k=0
        for lo,hi in zip(edges[:-1],edges[1:]):
            inside=a[(a>=lo)&(a<hi)]
            if len(inside)==1 and k<KMAX: PK[i,k]=inside[0];VL[i,k]=lo;VL[i,k+1]=hi;k+=1
        if k<KMAX: VL[i,k+1:]=VL[i,k]
    return PK,VL,PER
from metrics_timing import ev
def peak_phase_loss(p,sp):
    n=p.shape[-1];h=torch.zeros(n,device=p.device);h[0]=1;h[1:(n+1)//2]=2
    if n%2==0: h[n//2]=1
    fr=torch.fft.fftfreq(n,1/D.FS).abs().to(p.device);bm=((fr>=0.1)&(fr<=0.6)).float()
    an=torch.fft.ifft(torch.fft.fft(p,dim=-1)*h*bm,dim=-1)+1e-6;ph=torch.atan2(an.imag,an.real);w=sp.abs()
    return ((w*(1-sp*torch.cos(ph))).sum(-1)/(w.sum(-1)+1e-6)).mean()
if os.environ.get('GEOSPLIT','0')=='1': RTAG+='_geo'
if os.environ.get('GEODROP'): RTAG+='_gd'+os.environ['GEODROP'].replace(',','')
# CANDMASK=abcd (a,b,c,d in {0,1}) keeps/zeroes the four candidates C_L, C_G, T_L, T_G (envdphi channel pairs 0-1, 2-3, 4-5, 6-7)
# for the 15-subset study; geometry context and profile stay shared. Tag _m<mask>. '1111' is the ordinary dual model.
CANDMASK=os.environ.get('CANDMASK','')
if CANDMASK:
    assert re.fullmatch(r'[01]{4}',CANDMASK) and '1' in CANDMASK,'CANDMASK must be four 0/1 chars with at least one 1'
    assert RADAR=='both','CANDMASK is a dual-input mask; use RADAR=both'
    if CANDMASK!='1111': RTAG+=f'_m{CANDMASK}'          # extra tag, e.g. WTAG=_hop5 for weights trained on another grid (never overwrite the 6 s set)
HOP_S=getattr(D,'HOP_S',6.0)
START_S=np.zeros(N,np.float32)                           # each row's start time inside its subject's record (rank x hop)
for _u in np.unique(S):
    _q=np.flatnonzero(S==_u);START_S[_q]=np.arange(len(_q))*HOP_S
# CALSTATS=1 (diffusion_model.CALSTATS): AdaLN also receives c_u = (ampCV, intCV) of a random NON-overlapping 84 s belt
# block of the same subject, re-drawn every epoch, z-scored with the fold's training-subject statistics (buffers in
# the weights). Requires a WTAG so the plain _rc weights are never overwritten.
CALSTATS=D.CALSTATS
if CALSTATS:
    import calibration_stats as CS
    assert os.environ.get('WTAG',''),'CALSTATS=1 needs WTAG (e.g. WTAG=_cu) so existing weights are not overwritten'
# ---- episodic (FOMAML) pre-training and its compute-matched control (docs/crosscheck_2026-09-06/episodic_claude_critique.md, Stage 1)
# META=1: warm start from the fold EMA weights _envdphi_s2{INIT}_s{SEED}_f{fi}.pt (never overwritten), then META_OUT outer steps;
#   each outer step draws META_MB training subjects, adapts a COPY on that subject's support rows (the calibration block:
#   windows lying inside the first META_CAL s; META_SUP rows of them) for META_K AdamW steps at META_LR with the training loss
#   (= FH.adapt's loss), scores the adapted copy on META_Q query rows starting >= META_QMIN s, and applies the query-loss
#   gradient of the adapted copy to the initial weights (first-order MAML). AdamW META_LR, clip 1, EMA META_EMA; raw and EMA saved.
#   Diagnostics per outer step (CSV): ||theta_adapt-theta||/||theta||, cos(meta-gradient, plain gradient on the same query rows
#   and noise draw), adapted / un-adapted query loss, support loss after adaptation, and (every META_DIAG100 steps) the query loss
#   after 100 instead of META_K inner steps.
# CTL=1: same warm start, seed, lr and EMA, the standard loss on uniformly drawn training rows, batch 32, as many steps as the
#   meta run's sample passes: META_OUT*META_MB*(META_K*META_SUP+META_Q)/32.
META=os.environ.get('META','0')=='1';CTL=os.environ.get('CTL','0')=='1';assert not (META and CTL),'META and CTL are separate runs'
INIT=os.environ.get('INIT','')
META_K=int(os.environ.get('META_K','20'));META_LR=float(os.environ.get('META_LR','5e-5'));META_OUT=int(os.environ.get('META_OUT','200'))
META_MB=int(os.environ.get('META_MB','4'));META_Q=int(os.environ.get('META_Q','32'));META_SUP=int(os.environ.get('META_SUP','8'))
META_QMIN=float(os.environ.get('META_QMIN','84'));META_EMA=float(os.environ.get('META_EMA','0.98'));META_BLOCK=os.environ.get('META_BLOCK','head')
META_CAL=float(os.environ.get('META_CAL','84'));META_DIAG=os.environ.get('META_DIAG','1')=='1';META_DIAG100=int(os.environ.get('META_DIAG100','50'))
META_BS=int(os.environ.get('META_BS','32'))              # control-arm batch size (the reference training batch)
FOLDS=[int(v) for v in os.environ.get('FOLDS',','.join(map(str,range(NFOLD)))).split(',')]   # subset of folds to train (timing runs)
if META or CTL:
    assert os.environ.get('WTAG',''),'META/CTL need WTAG (e.g. WTAG=_rc_meta / _rc_ctl) so the warm-start weights are never overwritten'
    assert INIT and RTAG!=INIT,f'INIT={INIT!r} must name the warm-start weights and differ from the output tag {RTAG!r}'
    assert META_BLOCK in ('head','random')
def step_loss(net,j,TT,cut=None,t=None,eps=None):
    """the training loss of one mini-batch: j = fold-local row indices into the fold tensors TT (x, c, p, z, g, [sp, stv, pkv]).
    noise-MSE + zone CE + S2 (exactly FH.adapt's loss) plus the optional ALIGN / STAT / PEAK terms (all off by default).
    t and eps may be handed in so two nets are scored on the identical noise draw (meta-gradient diagnostics)."""
    xt,ct,pt,zt,gt=TT['x'],TT['c'],TT['p'],TT['z'],TT['g'];B=len(j)
    xj=batch_dropout(V1.rotate(xt[j]) if ROT else xt[j])
    tok,_,lz=net.cond(xj,ct[j],pt[j])
    if t is None: t=torch.randint(0,D.T_DIFF,(B,),device=dev)
    if eps is None: eps=torch.randn_like(gt[j])
    a=D.ab[t][:,None,None];xn=a.sqrt()*gt[j]+(1-a).sqrt()*eps
    e=net(xn,t,tok,cu=(cut[j] if cut is not None else None));l=Fn.mse_loss(e,eps)
    x0=(xn-(1-a).sqrt()*e)/a.sqrt()
    l=l+Fn.cross_entropy(lz.reshape(-1,14),zt[j].reshape(-1),ignore_index=-1)
    l=l+LS.loss_S2(x0,gt[j],B)
    if ALIGNLAM>0:
        e_=net.embed(xj);nch=e_.shape[2];nb=nch//2//2 if nch%4==0 else nch//2   # per radar block of nb bins x [env,dphi]
        blk=nch//2;pairs=[(c,c+2) for r0 in (0,blk) for c in range(r0,r0+blk-2)]
        l=l+ALIGNLAM*torch.stack([nn.functional.mse_loss(e_[:,:,a],e_[:,:,b]) for a,b in pairs]).mean()
    if STATLAM>0: STV=TT['stv'];l=l+STATLAM*stat_loss(x0.reshape(B,-1),STV[0][j],STV[1][j],STV[2][j])
    if PEAKLAM>0 and PEAKLOSS=='phase': l=l+PEAKLAM*peak_phase_loss(x0.reshape(B,-1),TT['sp'][j])
    if PEAKLAM>0 and PEAKLOSS=='adj':
        PKV=TT['pkv'];xf=x0.reshape(B,-1);l=l+PEAKLAM*(peak_adjust_loss(xf,*[a[j] for a in PKV[0]])+peak_adjust_loss(-xf,*[a[j] for a in PKV[1]]))
    if PEAKLAM>0 and PEAKLOSS=='lagdup':
        PKV=TT['pkv'];xf=x0.reshape(B,-1);l=l+PEAKLAM*(peak_lagdup_loss(xf,*[a[j] for a in PKV[0]])+peak_lagdup_loss(-xf,*[a[j] for a in PKV[1]]))
    return l
def episode_plan(tr_rows,rs=None):
    """per training subject: (support pool, query pool) of GLOBAL rows. head: support = rows whose 42 s window lies inside the
    first META_CAL s (START_S+42 <= META_CAL; 8 rows on the 6 s grid, 43 on hop 1), query = rows starting >= META_QMIN s
    (FH.ev_head's non-overlap rule). random: the calibration block starts at a random time b (re-drawn per call), support = windows
    inside [b, b+META_CAL], query = rows whose window does not overlap that block."""
    EPL={}
    for u in np.unique(S[tr_rows]):
        r=tr_rows[S[tr_rows]==u];st=START_S[r];L=float(st[-1]+42.0)
        if META_BLOCK=='head': b=0.0;sup=r[st+42.0<=META_CAL+1e-6];qry=r[st>=META_QMIN-1e-6]
        else:
            b=float(rs.rand()*max(0.0,L-META_CAL));sup=r[(st>=b-1e-6)&(st+42.0<=b+META_CAL+1e-6)];qry=r[(st+42.0<=b+1e-6)|(st>=b+META_CAL-1e-6)]
        if len(sup)>=1 and len(qry)>=1: EPL[int(u)]=(sup,qry)
    return EPL
def inner_adapt(net,j,k,lr,TT):
    """FH.adapt on a copy: k AdamW steps (fresh optimiser state, lr, weight decay 1e-4, clip 1) on the support batch j, all parameters"""
    a=copy.deepcopy(net);[p.requires_grad_(True) for p in a.parameters()];opt=torch.optim.AdamW(a.parameters(),lr,weight_decay=1e-4);a.train();l=torch.zeros(())
    for _ in range(k):
        l=step_loss(a,j,TT);opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(a.parameters(),1.);opt.step()
    return a,float(l.detach())
def _flat(gs): return torch.cat([g.reshape(-1) for g in gs])
def meta_outer(net,ema,opt,EPL,loc,TT,rs,step,diag):
    """one FOMAML outer step: META_MB subjects -> adapted copies -> query-loss gradients averaged onto the initial weights"""
    subs_=list(EPL);pick=rs.choice(len(subs_),META_MB,replace=False)
    P0=[p.detach() for p in net.parameters()];th_norm=_flat(P0).norm()
    mg=[torch.zeros_like(p) for p in net.parameters()];pg=[torch.zeros_like(p) for p in net.parameters()];rec=dict(rel=[],cos=[],lq=[],lp=[],ls=[],lq100=[])
    for q_,pi in enumerate(pick):
        sup,qry=EPL[subs_[pi]]
        s_rows=sup if len(sup)<=META_SUP else sup[rs.choice(len(sup),META_SUP,replace=False)]
        q_rows=qry if len(qry)<=META_Q else qry[rs.choice(len(qry),META_Q,replace=False)]
        js=torch.tensor(loc[s_rows],device=dev);jq=torch.tensor(loc[q_rows],device=dev)
        ad,ls=inner_adapt(net,js,META_K,META_LR,TT)
        t=torch.randint(0,D.T_DIFF,(len(jq),),device=dev);eps=torch.randn_like(TT['g'][jq])
        lq=step_loss(ad,jq,TT,t=t,eps=eps);ad.zero_grad(set_to_none=False);lq.backward()
        g_ad=[(p.grad if p.grad is not None else torch.zeros_like(p)).detach().clone() for p in ad.parameters()]   # parameters off the forward path (unused heads) have no gradient
        used=[p.grad is not None for p in ad.parameters()]
        for m,g in zip(mg,g_ad): m.add_(g,alpha=1.0/len(pick))
        rec['lq'].append(float(lq.detach()));rec['ls'].append(ls)
        if diag:
            with torch.no_grad(): rec['rel'].append(float(_flat([pa-p0 for pa,p0 in zip(ad.parameters(),P0)]).norm()/(th_norm+1e-12)))
            net.zero_grad(set_to_none=False);lp=step_loss(net,jq,TT,t=t,eps=eps);lp.backward()      # plain gradient at theta, same rows & noise
            g_pl=[(p.grad if p.grad is not None else torch.zeros_like(p)).detach().clone() for p in net.parameters()];fa=_flat(g_ad);fp=_flat(g_pl)
            for m,g in zip(pg,g_pl): m.add_(g,alpha=1.0/len(pick))
            rec['cos'].append(float((fa*fp).sum()/(fa.norm()*fp.norm()+1e-12)));rec['lp'].append(float(lp))
            if q_==0 and META_DIAG100>0 and step%META_DIAG100==0:
                ad100,_=inner_adapt(net,js,100,META_LR,TT)
                with torch.no_grad(): l100=step_loss(ad100,jq,TT,t=t,eps=eps)
                rec['lq100'].append(float(l100));del ad100
        del ad
    fm=_flat(mg);fg=_flat(pg);cos_agg=float((fm*fg).sum()/(fm.norm()*fg.norm()+1e-12)) if diag else np.nan
    net.zero_grad(set_to_none=False)
    for p,m,u in zip(net.parameters(),mg,used): p.grad=m if u else None                    # unused parameters stay grad-less (AdamW skips them, as in plain training)
    gn=float(nn.utils.clip_grad_norm_(net.parameters(),1.));opt.step()
    with torch.no_grad():
        for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(META_EMA).add_(pn,alpha=1-META_EMA)
    return dict(rel=np.mean(rec['rel']) if rec['rel'] else np.nan,cos=np.mean(rec['cos']) if rec['cos'] else np.nan,cos_agg=cos_agg,lq=np.mean(rec['lq']),
                lp=np.mean(rec['lp']) if rec['lp'] else np.nan,ls=np.mean(rec['ls']),lq100=(rec['lq100'][0] if rec['lq100'] else np.nan),gnorm=gn)
def cu_tables(subs_all):
    """per subject: (block starts s, block stats (nb,2)) for every 84 s block at a 1 s hop"""
    BT={};nfin=0;ntot=0
    for u in subs_all:
        st,v=CS.block_table(int(u));BT[int(u)]=(st,v);nfin+=int(np.isfinite(v).all(1).sum());ntot+=len(v)
    print(f'  calstats: {len(BT)} subjects, {ntot} blocks of {CS.CAL_S:g} s, {nfin} with a valid vector',flush=True)
    return BT
def cu_plan(BT,rows):
    """for every training row the indices of its subject's blocks that do NOT overlap the row's 42 s window"""
    VB=[]
    for r in rows:
        st,v=BT[int(S[r])];ok=CS.valid_blocks(st,float(START_S[r]))&np.isfinite(v).all(1);VB.append(np.flatnonzero(ok))
    return VB
def cu_draw(BT,rows,VB,rs,mu,sd):
    """one z-scored vector per training row from a uniformly drawn valid block; rows with no valid block get 0 (= mean)"""
    Z=np.zeros((len(rows),CS.CU_DIM),np.float32);miss=0
    for k,r in enumerate(rows):
        vb=VB[k]
        if len(vb)==0: miss+=1;continue
        raw=BT[int(S[r])][1][vb[int(rs.rand()*len(vb))]];Z[k]=(raw-mu)/sd
    return Z,miss

# GEOSPLIT=1 with RADAR=com|tv: also remove the other radar from the geometry context (12 per-chunk features:
# 0 r_com 1 r_tv 2 gap 3 |gap| 4 tanh(gap) 5 ratio 6 corner 7 v_com 8 v_tv 9 approach 10 gl_com 11 gl_tv)
# and from the range profile (COM half then TV half). Applied to the NORMALISED arrays, so zero = train mean.
GEOSPLIT=os.environ.get('GEOSPLIT','0')=='1'
def geo_mask(Cn,Pn,radar):
    if not GEOSPLIT or radar not in ('com','tv'): return Cn,Pn
    Cn=Cn.copy();Pn=Pn.copy();half=Pn.shape[-1]//2
    keep={'com':[0,7,10],'tv':[1,8,11]}[radar]
    drop=[k for k in range(12) if k not in keep];Cn[...,drop]=0.0
    if radar=='com': Pn[...,half:]=0.0
    else: Pn[...,:half]=0.0
    return Cn,Pn
def radar_mask(X):
    Y=X.copy()
    if RADAR=='com': Y[:,4:]=0.0          # candidates 2,3 = TV-LOS, TV-Ghost
    elif RADAR=='tv': Y[:,:4]=0.0         # candidates 0,1 = COM-LOS, COM-Ghost
    for c,keep in enumerate(CANDMASK or '1111'):
        if keep=='0': Y[:,2*c:2*c+2]=0.0   # candidate c = (C_L, C_G, T_L, T_G)[c]
    return Y
# RADAR=dropout: dual input, but per training sample one radar is zeroed with prob PDROP each
# (both kept otherwise). The model must stay useful on either radar alone - the fusion cannot lean on
# a joint pattern that only exists when both are present. Inference sees the dual input unmasked.
PDROP=float(os.environ.get('PDROP','0.25'))
GROUPAUX=int(os.environ.get('GROUPAUX','8'))   # RADAR=groups: channels >= GROUPAUX are the auxiliary group (Parallel in pv2S12)
def batch_dropout(x):
    if RADAR=='groups':
        # group dropout (ModDrop-style) so every condition used by two-scale CFG at inference is trained:
        #   p 0.10 null (all radar channels 0 - the CFG reference), p 0.20 aux dropped (= the base condition),
        #   p 0.10 one radar's LOS dropped (so LOS-absent, aux-present inputs are in-distribution); else full
        B=x.shape[0];u=torch.rand(B,device=x.device);y=x.clone()
        y[u<0.10]=0.0
        y[(u>=0.10)&(u<0.30),GROUPAUX:]=0.0
        y[(u>=0.30)&(u<0.35),0:2]=0.0             # COM LOS [|z|, dphi]
        y[(u>=0.35)&(u<0.40),4:6]=0.0             # TV LOS
        return y
    if RADAR!='dropout': return x
    B=x.shape[0];u=torch.rand(B,device=x.device)
    y=x.clone()
    y[u<PDROP,:4]=0.0                          # drop COM
    y[(u>=PDROP)&(u<2*PDROP),4:]=0.0           # drop TV
    return y

def _load_input():
    """model input cache: envdphi by default; INCACHE=<name> loads preprocessed/_<name>{GTAG}.npy instead
    (e.g. INCACHE=mc_p0 = per candidate [integrated phase track band-passed 0.1-0.6 Hz, amplitude track] - the
    'author phase channel' family). Shape (N,8,W) or (N,8,NC,CH); returned as (N,8,NC,CH) float32."""
    name=os.environ.get('INCACHE','')
    if not name: return RA.envdphi_cache()
    X=np.load(f'{PRE}/_{name}{V1.GTAG}.npy').astype(np.float32);nch=X.size//(len(X)*NC*CH);X=X.reshape(len(X),nch,NC,CH)
    V1.NCH=nch;V1.NCAND=nch//2                   # the encoder's channel count follows the cache (e.g. envdphi_ov = 16)
    print(f'  input cache: _{name}{V1.GTAG}.npy {X.shape}  NCH={nch}',flush=True);return X
if __name__=='__main__':
    set_det(1)
    subs=np.array([u for u in np.unique(S) if u not in EXCL])
    pmx=np.random.RandomState(0).permutation(len(subs))
    folds=[subs[pmx[i::NFOLD]] for i in range(NFOLD)]
    ED=radar_mask(_load_input())
    Gt2=G.reshape(N,NC,CH).astype(np.float32)
    SP=np.zeros((N,NC*CH),np.float32)
    if PEAKLAM>0:
        for i in range(N): SP[i,ev(G[i],1).astype(int)]=1.0;SP[i,ev(G[i],-1).astype(int)]=-1.0
    STT=stat_tables() if STATLAM>0 else None
    PKT=event_tables(1) if PEAKLAM>0 and PEAKLOSS in ('adj','lagdup') else None;VLT=event_tables(-1) if PKT is not None else None
    print(f'  seed {SEED}  EP {EP}  align={ALIGNLAM}  ema={EMA_DECAY}  peaklam={PEAKLAM} ({PEAKLOSS})  radar={RADAR}  candmask={CANDMASK or "1111"}  randcrop={RANDCROP} (hop {HOP_S:g} s)  calstats={CALSTATS}  saving fold EMA weights to {OUTD}',flush=True)
    BT=cu_tables(subs) if CALSTATS else None
    if META or CTL:
        import time,csv
        n_ctl=int(round(META_OUT*META_MB*(META_K*META_SUP+META_Q)/META_BS))
        print(f'  {"META (FOMAML)" if META else "CTL (plain)"}: warm start _envdphi_s2{INIT}_s{SEED}_f*.pt -> _envdphi_s2{RTAG}_s{SEED}_f*.pt (+ _raw); '
              +(f'K={META_K} lr={META_LR:g} outer={META_OUT} mb={META_MB} sup={META_SUP} q={META_Q} qmin={META_QMIN:g}s cal={META_CAL:g}s block={META_BLOCK} ema={META_EMA} diag={META_DIAG}' if META
                else f'{n_ctl} steps x batch {META_BS} = {n_ctl*META_BS} samples (= {META_OUT}x{META_MB}x({META_K}x{META_SUP}+{META_Q})) lr={META_LR:g} ema={META_EMA}'),flush=True)
        dcsv=f'{OUTD}/_diag{RTAG}_s{SEED}.csv';dfh=open(dcsv,'w',newline='');dw=csv.writer(dfh)
        dw.writerow(['fold','step','rel_move','cos_meta_plain','cos_agg','loss_q_adapted','loss_q_plain','loss_sup_post','loss_q_adapt100','gnorm','sec'] if META else ['fold','step','loss','sec']);dfh.flush()
    for fi,te in enumerate(folds):
        if fi not in FOLDS: continue
        trm=~np.isin(S,te)&~np.isin(S,EXCL)
        Cn,Pn=D.fold_norm(trm);Cn,Pn=geo_mask(Cn,Pn,os.environ.get('GEORADAR') or RADAR);   # GEORADAR: strict single radar for a cache that already holds only that radar's channels
        if os.environ.get('GEODROP'): Cn=Cn.copy();Cn[...,[int(v) for v in os.environ['GEODROP'].split(',')]]=0.0   # e.g. 10,11 = LOS/SR energy ratio
        T_=lambda a,m:torch.tensor(a[m],device=dev)
        ct,pt,zt,gt,xt=T_(Cn,trm),T_(Pn,trm),T_(ZONE,trm),T_(Gt2,trm),T_(ED,trm);spt=T_(SP,trm);STV=(None if STT is None else [T_(a,trm) for a in STT]);PKV=None if PKT is None else [[T_(a,trm) for a in PKT],[T_(a,trm) for a in VLT]]
        TT=dict(x=xt,c=ct,p=pt,z=zt,g=gt,sp=spt,stv=STV,pkv=PKV)
        torch.manual_seed(900+fi+1000*SEED);net=V1.Net().to(dev)
        if META or CTL:
            net.load_state_dict(torch.load(f'{OUTD}/_envdphi_s2{INIT}_s{SEED}_f{fi}.pt',map_location=dev))
            tr_rows=np.flatnonzero(trm);loc=np.full(N,-1);loc[tr_rows]=np.arange(len(tr_rows))
            ema=copy.deepcopy(net);[p.requires_grad_(False) for p in ema.parameters()]
            opt=torch.optim.AdamW(net.parameters(),META_LR,weight_decay=1e-4);rs=np.random.RandomState(4241*SEED+fi);t0=time.time()
            if META:
                EPL=episode_plan(tr_rows,rs);ns=[len(v[0]) for v in EPL.values()];nq=[len(v[1]) for v in EPL.values()]
                print(f'  fold{fi}: {len(EPL)} training subjects, support pool {min(ns)}..{max(ns)} rows, query pool {min(nq)}..{max(nq)} rows',flush=True)
                for step in range(META_OUT):
                    if META_BLOCK=='random': EPL=episode_plan(tr_rows,rs)
                    net.train();r=meta_outer(net,ema,opt,EPL,loc,TT,rs,step,META_DIAG);el=time.time()-t0
                    dw.writerow([fi,step,f'{r["rel"]:.3e}',f'{r["cos"]:.4f}',f'{r["cos_agg"]:.4f}',f'{r["lq"]:.4f}',f'{r["lp"]:.4f}',f'{r["ls"]:.4f}',f'{r["lq100"]:.4f}',f'{r["gnorm"]:.3f}',f'{el:.1f}']);dfh.flush()
                    if step%10==0 or step==META_OUT-1: print(f'  fold{fi} step {step:4d}  rel {r["rel"]:.2e}  cos {r["cos"]:.3f} (agg {r["cos_agg"]:.3f})  q_adapt {r["lq"]:.4f}  q_plain {r["lp"]:.4f}  sup {r["ls"]:.4f}  q_adapt100 {r["lq100"]:.4f}  {el/((step+1)*META_MB):.2f} s/episode',flush=True)
            else:
                n=len(xt);idx=torch.empty(0,dtype=torch.long,device=dev)
                for step in range(n_ctl):
                    if len(idx)<META_BS: idx=torch.cat([idx,torch.randperm(n,device=dev)])
                    j=idx[:META_BS];idx=idx[META_BS:];net.train()
                    l=step_loss(net,j,TT);opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                    with torch.no_grad():
                        for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(META_EMA).add_(pn,alpha=1-META_EMA)
                    if step%50==0 or step==n_ctl-1: dw.writerow([fi,step,f'{float(l):.4f}',f'{time.time()-t0:.1f}']);dfh.flush()
                    if step%500==0 or step==n_ctl-1: print(f'  fold{fi} step {step:5d}/{n_ctl}  loss {float(l):.4f}  {time.time()-t0:.0f} s',flush=True)
            ema.eval();net.eval()
            torch.save(ema.state_dict(),f'{OUTD}/_envdphi_s2{RTAG}_s{SEED}_f{fi}.pt');torch.save(net.state_dict(),f'{OUTD}/_envdphi_s2{RTAG}_raw_s{SEED}_f{fi}.pt')
            del net,ema;torch.cuda.empty_cache();print(f'  fold{fi} saved ({time.time()-t0:.0f} s)',flush=True);continue
        if CALSTATS:
            tr_rows=np.flatnonzero(trm);tr_subs=[u for u in subs if u not in te]
            allv=np.concatenate([BT[int(u)][1] for u in tr_subs]);allv=allv[np.isfinite(allv).all(1)]
            cu_mu=allv.mean(0).astype(np.float32);cu_sd=(allv.std(0)+1e-6).astype(np.float32)   # TRAINING subjects only
            with torch.no_grad(): net.cu_mu.copy_(torch.tensor(cu_mu));net.cu_sd.copy_(torch.tensor(cu_sd))
            VB=cu_plan(BT,tr_rows);cu_rs=np.random.RandomState(7919*SEED+fi)
            print(f'  fold{fi} calstats mu {cu_mu} sd {cu_sd} (from {len(tr_subs)} training subjects, {len(allv)} blocks); rows without a valid block {sum(len(v)==0 for v in VB)}/{len(VB)}',flush=True)
        ema=copy.deepcopy(net);[p.requires_grad_(False) for p in ema.parameters()]
        opt=torch.optim.AdamW(net.parameters(),2e-4,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP);n=len(xt)
        n_ep=int(round(n*HOP_S/6.0)) if RANDCROP else n
        for ep in range(EP):
            net.train();idx=torch.randperm(n,device=dev)[:n_ep]
            if CALSTATS: cut=torch.tensor(cu_draw(BT,tr_rows,VB,cu_rs,cu_mu,cu_sd)[0],device=dev)   # fresh block per row, every epoch
            for b in range(0,len(idx),32):
                j=idx[b:b+32]
                l=step_loss(net,j,TT,cut=(cut if CALSTATS else None))
                opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                with torch.no_grad():
                    for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(EMA_DECAY).add_(pn,alpha=1-EMA_DECAY)
            sch.step()
        ema.eval()                                     # loaders must also call .eval() after load
        torch.save(ema.state_dict(),f'{OUTD}/_envdphi_s2{RTAG}_s{SEED}_f{fi}.pt')
        del net,ema;torch.cuda.empty_cache();print(f'  fold{fi} saved',flush=True)
    print('  done')
