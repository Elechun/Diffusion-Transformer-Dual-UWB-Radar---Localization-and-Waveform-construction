"""Five structural changes aimed at making the waveform LOOK like the belt, each switchable on its own.

    A  a loss that tolerates small timing error - per-chunk banded soft-DTW plus a shift-tolerant
       correlation. Peaks that are right but 0.4 s late stop being punished as if they were absent, so the
       model has no reason to smear them out
    B  the target becomes the breathing PHASE. The generator emits (cos p, sin p) and the waveform is read
       off the angle, so what comes out is an oscillator by construction. This is the phase of the
       respiration signal, not the radar carrier - nothing to do with the two radars being non-coherent
    C  the gate stops picking and starts mixing: it predicts eight real coefficients per chunk, supervised
       directly by the ridge least-squares fit of the eight channels to the belt. That is 8 numbers per
       chunk of supervision instead of one scalar per 42 s window
    D  the four candidates are time-aligned to each other before they are fused. No belt is involved - they
       are the same breath seen from two angles, so they align against each other. Averaging misaligned
       signals cancels peaks, which is why the plain four-way average was the worst arm
    E  the candidate encoder becomes a dilated stack with an 11 s receptive field. At 15 breaths a minute one
       cycle is 4 s, so the current 3 s encoder has never once seen a whole breath

Training is what gets the timing slack. Scoring does not: breath F1 and |corr| are computed at zero lag
exactly as before, so a lenient loss cannot buy a lenient number.

Every arm also runs through a donor - another subject's radar through the same weights - so the prior-driven
part stays visible even while the waveform is the thing being looked at.
"""
import _path  # noqa: F401  (see _path.py)
import os,math,copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
from scipy.signal import hilbert
from scipy.stats import ttest_rel
from normalize import zn
from metrics import score
import diffusion_model as D
import signal_processing as RC
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed'
GEO=os.environ.get('GEO','_geo_dataset_fc729.npz');GTAG=GEO.replace('_geo_dataset','').replace('.npz','')
dev=D.dev;NC,CH,FS,DM,W=D.NC,D.CH,D.FS,D.DM,D.NC*D.CH
PAIRX=os.environ.get('PAIRX','0')=='1'   # LOS<->Ghost cross-reading inside each radar before pooling
EP=int(os.environ.get('EP','60'));GS=3.0;PHI=0.7;EMA_DECAY=0.999;SEED=int(os.environ.get('SEED','0'))
NFOLD=int(os.environ.get('NFOLD','5'))
S,G,Gt,ZONE=D.S,D.G,D.Gt,D.ZONE;N=len(D.X)
XIQ0=np.load(f'{PRE}/_iq_candidates{GTAG}.npy').astype(np.float32)      # (N,8,NC,CH)
NCH=XIQ0.shape[1];NCAND=NCH//2
MAXLAG=int(round(0.55*FS))                                              # +-0.55 s, about a tenth of a breath
# ---------------------------------------------------------------------------------------------------
# D - align the candidates to each other, per chunk, without the belt
# ---------------------------------------------------------------------------------------------------
def _zc(a):
    """z-score along the last axis"""
    a=a-a.mean(-1,keepdims=True);return a/(a.std(-1,keepdims=True)+1e-9)
def _cut(full,off):
    """chunk-sized slices taken at a per-chunk offset -> (n, NC, CH); offsets that fall off the end clamp"""
    base=(np.arange(NC)[:,None]*CH+np.arange(CH)[None,:])
    idx=np.clip(base[None]+off[:,:,None],0,W-1)
    return np.take_along_axis(full[:,None,:].repeat(NC,1),idx,axis=2)
def align_candidates(xiq):
    """medoid candidate anchors each chunk; the others are re-cut at the lag that matches it best.

    Only the candidates are compared - the belt is never touched. Vectorised over windows and chunks, so the
    whole dataset is one pass over 2*MAXLAG+1 offsets instead of a loop over every chunk.
    """
    n=len(xiq);full=xiq.reshape(n,NCH,W)
    seg0=np.stack([_zc(full[:,2*c].reshape(n,NC,CH)) for c in range(NCAND)],2)      # (n,NC,4,CH)
    Cm=np.abs(np.einsum('nqac,nqbc->nqab',seg0,seg0)/CH)
    for c in range(NCAND): Cm[:,:,c,c]=0
    anc=Cm.sum(-1).argmax(-1)                                                        # (n,NC)
    A=np.take_along_axis(seg0,anc[:,:,None,None].repeat(CH,3),axis=2)[:,:,0]         # (n,NC,CH)
    lags=np.zeros((n,NC,NCAND),np.int16);out=np.empty_like(xiq)
    for c in range(NCAND):
        sc=np.full((n,NC,2*MAXLAG+1),-9.,np.float32)
        for li,L in enumerate(range(-MAXLAG,MAXLAG+1)):
            ok=(np.arange(NC)*CH+L>=0)&(np.arange(NC)*CH+L+CH<=W)
            v=_zc(_cut(full[:,2*c],np.full((n,NC),L)))
            s=np.abs((v*A).sum(-1)/CH)
            sc[:,:,li]=np.where(ok[None],s,-9.)
        L=sc.argmax(-1).astype(np.int16)-MAXLAG
        L=np.where(anc==c,0,L).astype(np.int16);lags[:,:,c]=L
        out[:,2*c]=_cut(full[:,2*c],L.astype(int));out[:,2*c+1]=_cut(full[:,2*c+1],L.astype(int))
    return out,lags
# ---------------------------------------------------------------------------------------------------
# C - the mixture the belt would have asked for, per chunk (ridge, so eight coefficients stay tame)
# ---------------------------------------------------------------------------------------------------
def ls_weights(xiq,lam=0.5):
    A=xiq.transpose(0,2,3,1).reshape(-1,CH,NCH).astype(np.float64)      # (N*NC, CH, 8)
    A=A/(A.std(1,keepdims=True)+1e-6)
    b=Gt.reshape(-1,CH,1).astype(np.float64)
    b=(b-b.mean(1,keepdims=True))/(b.std(1,keepdims=True)+1e-9)
    G_=np.einsum('nki,nkj->nij',A,A)+lam*CH*np.eye(NCH)[None]
    r=np.einsum('nki,nkj->nij',A,b)
    w=np.linalg.solve(G_,r)[:,:,0]
    w=w/(np.linalg.norm(w,axis=1,keepdims=True)+1e-9)
    return w.reshape(N,NC,NCH).astype(np.float32)
# ---------------------------------------------------------------------------------------------------
# A - losses that forgive a small shift
# ---------------------------------------------------------------------------------------------------
def rho_shift(a,b,maxlag=MAXLAG,tau=.02):
    """correlation at the best small shift, softened so it stays differentiable"""
    outs=[]
    for L in range(-maxlag,maxlag+1):
        if L<0: x,y=a[:,-L:],b[:,:L]
        elif L>0: x,y=a[:,:-L],b[:,L:]
        else: x,y=a,b
        outs.append(D.rho(x,y))
    r=torch.stack(outs,-1)
    return (torch.softmax(r/tau,-1)*r).sum(-1)
DEC=3                                    # the band stops at 0.6 Hz, so 17/3 Hz is still nine times Nyquist
def sdtw(x,y,gamma=.1,band=max(1,round(MAXLAG/DEC)),dec=DEC):
    """banded soft-DTW, solved one anti-diagonal at a time so the whole batch of 3 s chunks moves together.

    Two rolling vectors hold the previous two diagonals, so the T x T table is never materialised for the
    recursion and nothing is written in place. The pair is average-pooled first: the signal carries nothing
    above 0.6 Hz, so the shape is untouched and the quadratic table shrinks ninefold.
    """
    if dec>1:
        x=Fn.avg_pool1d(x.unsqueeze(1),dec,dec).squeeze(1)
        y=Fn.avg_pool1d(y.unsqueeze(1),dec,dec).squeeze(1)
    B,T=x.shape;dv=x.device
    x=(x-x.mean(1,keepdim=True))/(x.std(1,keepdim=True)+1e-6)
    y=(y-y.mean(1,keepdim=True))/(y.std(1,keepdim=True)+1e-6)
    dm=(x[:,:,None]-y[:,None,:]).pow(2)
    ii=torch.arange(T,device=dv)
    dm=dm.masked_fill(((ii[:,None]-ii[None,:]).abs()>band)[None],1e4)
    INF=1e9
    Dm2=torch.full((B,T+1),INF,device=dv)                    # diagonal k-2
    Dm1=torch.full((B,T+1),INF,device=dv);Dm1=Dm1.clone();Dm1[:,0]=0.   # diagonal k-1 == k=0, R[0,0]=0
    for k in range(1,2*T+1):
        lo=max(1,k-T);hi=min(T,k-1)
        if lo>hi:
            Dm2,Dm1=Dm1,torch.full((B,T+1),INF,device=dv);continue
        i=torch.arange(lo,hi+1,device=dv);j=k-i
        prev=torch.stack([Dm1[:,i-1],Dm1[:,i],Dm2[:,i-1]],-1)
        sm=-gamma*torch.logsumexp(-prev/gamma,-1)
        val=dm[:,i-1,j-1]+sm
        Dk=torch.full((B,T+1),INF,device=dv).index_copy(1,i,val)
        Dm2,Dm1=Dm1,Dk
    return Dm1[:,T]/T
# ---------------------------------------------------------------------------------------------------
# E - dilated candidate encoder, receptive field about 11 s
# ---------------------------------------------------------------------------------------------------
class Dil(nn.Module):
    def __init__(s,dm=DM,c=48):
        super().__init__()
        L=[nn.Conv1d(1,c,7,padding=3),nn.GELU()]
        for d in (1,2,4,8,16):
            L+= [nn.Conv1d(c,c,7,padding=3*d,dilation=d),nn.GELU()]
        s.f=nn.Sequential(*L);s.o=nn.Linear(2*c,dm)
    def forward(s,x):                                                   # x (B*NCH, 1, W)
        h=s.f(x).reshape(x.shape[0],-1,NC,CH)
        return s.o(torch.cat([h.mean(-1),h.max(-1).values],1).permute(0,2,1))   # (B*NCH, NC, dm)
# ---------------------------------------------------------------------------------------------------
class Net(D.DiT):
    def __init__(s,use_E=False,use_C=False,phase=False,dm=DM):
        super().__init__(dm)
        s.use_E,s.use_C,s.phase=use_E,use_C,phase
        s.cenc=nn.Sequential(nn.Conv1d(1,32,7,padding=3),nn.GELU(),nn.Conv1d(32,64,5,padding=2),nn.GELU(),
                             nn.Flatten(),nn.Linear(64*CH,dm)).to(dev)
        s.dil=Dil(dm).to(dev) if use_E else None
        # PAIRX: inside one radar the LOS bin and the Ghost bin see the same chest from two geometries. Let them read each
        # other before anything is pooled: per radar the two candidates' embeddings (env+dphi each) go through a small
        # mixer whose output is appended to the concatenation, so the model can use agreement/disagreement between them.
        s.pairx=PAIRX
        s.pair=nn.Sequential(nn.Linear(4*dm,dm),nn.GELU(),nn.Linear(dm,dm)).to(dev) if PAIRX else None
        s.cat=nn.Sequential(nn.Linear(NCH*dm+(2*dm if PAIRX else 0),dm),nn.GELU(),nn.Linear(dm,dm)).to(dev)
        s.mix=nn.Sequential(nn.Linear(2*dm,dm),nn.GELU(),nn.Linear(dm,1)).to(dev)
        s.cproj=nn.Sequential(nn.Linear(2*dm+14,dm),nn.GELU(),nn.Linear(dm,dm)).to(dev)
        if phase:
            s.xin=nn.Linear(2*CH,dm).to(dev)
            s.out=nn.Linear(dm,2*CH).to(dev);nn.init.zeros_(s.out.weight);nn.init.zeros_(s.out.bias)
    def embed(s,x):
        B=x.shape[0]
        if s.use_E:
            e=s.dil(x.reshape(B*NCH,1,W)).reshape(B,NCH,NC,-1).permute(0,2,1,3)
        else:
            e=s.cenc(x.reshape(B*NCH*NC,1,CH)).reshape(B,NCH,NC,-1).permute(0,2,1,3)
        return e                                                        # (B,NC,NCH,dm)
    def cond(s,x,c,pr):
        B=x.shape[0];e=s.embed(x);g=s.gctx(c);lz=s.loc(s.penc(pr))
        wv=None
        if s.use_C:
            wv=s.mix(torch.cat([e,g.unsqueeze(2).expand(-1,-1,NCH,-1)],-1)).squeeze(-1)
            wv=wv/(wv.norm(dim=-1,keepdim=True)+1e-6)                   # a direction, not a gain
            f=s.cat((e*wv[...,None]*math.sqrt(NCH)).reshape(B,NC,-1))
        else:
            fe=e.reshape(B,NC,-1)
            if s.pairx:                                   # channels: candidate c uses (2c,2c+1); radar r pairs candidates 2r,2r+1
                px=[]
                for r in range(2):
                    c0,c1=2*r,2*r+1
                    if 2*c1+1>=e.shape[2]: break
                    q=torch.cat([e[:,:,2*c0],e[:,:,2*c0+1],e[:,:,2*c1],e[:,:,2*c1+1]],-1)
                    px.append(s.pair(q))
                if px: fe=torch.cat([fe]+px,-1)
            f=s.cat(fe)
        return s.cproj(torch.cat([f,g,lz],-1)),wv,lz
# ---------------------------------------------------------------------------------------------------
# XATTN: radar-conditioning topology experiment (2026-09-10). The radar is NOT added to the waveform tokens;
# each candidate (COM-LOS, COM-Ghost, TV-LOS, TV-Ghost) of each 3 s chunk stays a separate token with a chunk-time
# and a candidate-identity embedding (NC x NCAND = 56 tokens), and every DiT block reads them by cross-attention
# (Q = waveform tokens, K/V = radar tokens) after its self-attention. Static per-chunk context (12 geometry
# features + zone logits) is still added to the waveform tokens and averaged into AdaLN.
#   XATTN=b : the radar tokens' mean is ALSO added to the AdaLN vector (global radar path kept)
#   XATTN=c : AdaLN carries only timestep + static context (dynamic radar reaches the waveform through cross-attention only)
# Loss, sampler, CFG (radar input zeroed -> unconditional tokens), few-shot and zone head are unchanged: cond() still
# returns (tok, None, lz) and forward(xt, t, tok) splits tok = [NC static tokens | NC*NCAND radar tokens].
XATTN=os.environ.get('XATTN','')
assert XATTN in ('','b','c'),'XATTN = b | c'
class XBlk(nn.Module):
    def __init__(s,dm=DM,nh=D.NHEAD):
        super().__init__()
        s.n1=nn.LayerNorm(dm,elementwise_affine=False);s.att=nn.MultiheadAttention(dm,nh,batch_first=True)
        s.nx=nn.LayerNorm(dm,elementwise_affine=False);s.xatt=nn.MultiheadAttention(dm,nh,batch_first=True)
        s.n2=nn.LayerNorm(dm,elementwise_affine=False)
        s.mlp=nn.Sequential(nn.Linear(dm,dm*4),nn.GELU(),nn.Linear(dm*4,dm))
        s.ada=nn.Sequential(nn.SiLU(),nn.Linear(dm,9*dm));nn.init.zeros_(s.ada[1].weight);nn.init.zeros_(s.ada[1].bias)
    def forward(s,x,c,R):
        p=s.ada(c).chunk(9,-1);sh1,sc1,g1,shx,scx,gx,sh2,sc2,g2=[q.unsqueeze(1) for q in p]
        h=s.n1(x)*(1+sc1)+sh1;a,_=s.att(h,h,h);x=x+g1*a                 # waveform self-attention (as before)
        h=s.nx(x)*(1+scx)+shx;a,_=s.xatt(h,R,R);x=x+gx*a                # waveform queries the radar tokens
        h=s.n2(x)*(1+sc2)+sh2;x=x+g2*s.mlp(h)
        return x
class XNet(D.DiT):
    def __init__(s,dm=DM):
        super().__init__(dm)
        s.rproj=nn.Linear(2*dm,dm)                                        # [envelope, dphi] embeddings -> one candidate token
        s.remb=nn.Parameter(torch.randn(1,1,NCAND,dm)*.02)                # candidate identity (COM-LOS, COM-Ghost, TV-LOS, TV-Ghost)
        s.rpos=nn.Parameter(torch.randn(1,NC,1,dm)*.02)                   # chunk time
        s.rnorm=nn.LayerNorm(dm)
        s.sproj=nn.Sequential(nn.Linear(dm+14,dm),nn.GELU(),nn.Linear(dm,dm))   # static per-chunk token: context + zone logits
        s.blocks=nn.ModuleList([XBlk(dm) for _ in range(D.NBLK)])
    def cond(s,x,c,pr):
        B=x.shape[0]
        e=s.cenc(x.reshape(B*NCH*NC,1,CH)).reshape(B,NCH,NC,-1).permute(0,2,1,3)       # (B,NC,NCH,dm); channels 2k,2k+1 = candidate k
        R=s.rproj(e.reshape(B,NC,NCAND,-1))+s.remb+s.rpos                             # (B,NC,NCAND,dm)
        R=s.rnorm(R).reshape(B,NC*NCAND,-1)
        g=s.gctx(c);lz=s.loc(s.penc(pr))
        st=s.sproj(torch.cat([g,lz],-1))                                                # (B,NC,dm), no radar inside
        return torch.cat([st,R],1),None,lz
    def forward(s,xt,t,tok,cu=None):
        assert cu is None,'XNet has no calibration-statistics path'
        st=tok[:,:NC];R=tok[:,NC:]
        h=s.xin(xt)+s.pos+st
        c=s.tmlp(D.temb(t))+st.mean(1)+(R.mean(1) if XATTN=='b' else 0.0)
        for b in s.blocks: h=b(h,c,R)
        return s.out(s.nout(h))
if XATTN: Net=XNet
def rotate(x):
    B=x.shape[0];th=torch.rand(B,1,1,device=x.device)*2*math.pi
    c,s_=torch.cos(th),torch.sin(th);y=x.clone()
    for k in range(0,x.shape[1],2):
        I=x[:,k];Q=x[:,k+1];y[:,k]=c*I-s_*Q;y[:,k+1]=s_*I+c*Q
    return y
def band_any(x,nchan):
    B=x.shape[0];v=x.reshape(B,NC,nchan,CH).permute(0,2,1,3).reshape(B*nchan,-1)
    v=torch.fft.irfft(torch.fft.rfft(v)*D.BAND,n=W)
    return v.reshape(B,nchan,NC,CH).permute(0,2,1,3).reshape(B,NC,nchan*CH)
@torch.no_grad()
def gen(net,tok,tku,n,nchan,seed=7,steps=50):
    torch.manual_seed(seed);x=torch.randn(n,NC,nchan*CH,device=dev)
    ts=torch.linspace(D.T_DIFF-1,0,steps).long().to(dev)
    for i in range(steps):
        t=ts[i].repeat(n);ec=net(x,t,tok);eu=net(x,t,tku);e=eu+GS*(ec-eu)
        a=D.ab[ts[i]];x0=band_any((x-(1-a).sqrt()*e)/a.sqrt(),nchan)
        x0c=band_any((x-(1-a).sqrt()*ec)/a.sqrt(),nchan)
        r=(x0c.reshape(n,-1).std(1)/(x0.reshape(n,-1).std(1)+1e-9)).reshape(n,1,1)
        x0=PHI*(x0*r)+(1-PHI)*x0;e=(x-a.sqrt()*x0)/(1-a).sqrt()
        if i<steps-1: a2=D.ab[ts[i+1]];x=a2.sqrt()*x0+(1-a2).sqrt()*e
        else: x=x0
    return x
def to_wave(x,phase):
    """model output -> one real waveform per window"""
    n=x.shape[0]
    if not phase: return x.reshape(n,-1)
    v=x.reshape(n,NC,2,CH).permute(0,2,1,3).reshape(n,2,-1)
    c,s_=v[:,0],v[:,1]
    return c/(torch.sqrt(c*c+s_*s_)+1e-6)                                # cos of the predicted angle
if __name__=='__main__':
    print('  preparing targets and aligned candidates ...',flush=True)
    an=f'{PRE}/_aligned_iq{GTAG}.npy'
    if os.path.exists(an): XIQ_A=np.load(an)
    else:
        XIQ_A,_=align_candidates(XIQ0);np.save(an,XIQ_A)
    LSW=ls_weights(XIQ0)                                                 # (N,NC,8)
    ph=np.angle(hilbert(G.astype(np.float64),axis=-1))
    GP=np.stack([np.cos(ph),np.sin(ph)],1).reshape(N,2,NC,CH).transpose(0,2,1,3) \
        .reshape(N,NC,2*CH).astype(np.float32)                           # (N,NC,2*CH)
    print(f'  aligned {XIQ_A.shape}   ls weights {LSW.shape}   phase target {GP.shape}',flush=True)
    ARMS=[('base (current)',{}),
          ('A  timing-tolerant loss',{'A':1}),
          ('E  dilated 11 s encoder',{'E':1}),
          ('A+E',{'A':1,'E':1}),
          ('D  aligned candidates',{'Dg':1}),
          ('C  mixture supervision',{'C':1}),
          ('A+C+D+E  everything',{'A':1,'C':1,'Dg':1,'E':1}),
          ('B  phase domain + all',{'A':1,'C':1,'Dg':1,'E':1,'P':1})]
    only=os.environ.get('ONLY')
    if only: ARMS=[a for a in ARMS if a[0].startswith(only)]
    subs=np.array([u for u in np.unique(S) if u not in (2,10,13,21)])
    pmx=np.random.RandomState(0).permutation(len(subs));folds=[subs[pmx[i::5]] for i in range(NFOLD)]
    if NFOLD<5: subs=np.sort(np.concatenate(folds))          # a short run only scores what it generated
    EV=np.concatenate([np.flatnonzero(S==u)[::7] for u in subs])
    rs=np.random.RandomState(1);DON=np.array([rs.choice(np.flatnonzero(S!=S[i])) for i in range(N)])
    OUT={a[0]:np.zeros((N,W),np.float32) for a in ARMS};DNR={a[0]:np.zeros((N,W),np.float32) for a in ARMS}
    for fi,te in enumerate(folds):
        trm=~np.isin(S,te)&~np.isin(S,[2,10,13,21])
        Cn,Pn=D.fold_norm(trm);T_=lambda a,m:torch.tensor(a[m],device=dev)
        ct,pt,zt=T_(Cn,trm),T_(Pn,trm),T_(ZONE,trm)
        gtw=T_(Gt.reshape(N,NC,CH),trm);gtp=T_(GP,trm);lsw=T_(LSW,trm)
        for nm,fl in ARMS:
            P=bool(fl.get('P'));nchan=2 if P else 1
            xt=T_(XIQ_A if fl.get('Dg') else XIQ0,trm)
            gt=gtp if P else gtw.reshape(-1,NC,CH)
            torch.manual_seed(900+fi+1000*SEED)
            net=Net(use_E=bool(fl.get('E')),use_C=bool(fl.get('C')),phase=P).to(dev)
            ema=copy.deepcopy(net);[p.requires_grad_(False) for p in ema.parameters()]
            opt=torch.optim.AdamW(net.parameters(),2e-4,weight_decay=1e-4)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP);n=len(xt)
            for ep in range(EP):
                net.train();idx=torch.randperm(n,device=dev)
                for b in range(0,n,32):
                    j=idx[b:b+32];B=len(j)
                    tok,wv,lz=net.cond(rotate(xt[j]),ct[j],pt[j])
                    t=torch.randint(0,D.T_DIFF,(B,),device=dev)
                    g_=gt[j].reshape(B,NC,nchan*CH);eps=torch.randn_like(g_)
                    a=D.ab[t][:,None,None];xn=a.sqrt()*g_+(1-a).sqrt()*eps
                    e=net(xn,t,tok);l=Fn.mse_loss(e,eps)
                    x0=(xn-(1-a).sqrt()*e)/a.sqrt()
                    l=l+Fn.cross_entropy(lz.reshape(-1,14),zt[j].reshape(-1),ignore_index=-1)
                    xf=x0.reshape(B,-1);gf=g_.reshape(B,-1)
                    if fl.get('A'):
                        l=l+0.3*sum((1-rho_shift(x0[:,:q+1].reshape(B,-1),g_[:,:q+1].reshape(B,-1))).mean()
                                    for q in range(1,NC))/(NC-1)
                        l=l+0.6*(1-rho_shift(xf,gf)).mean()
                        l=l+0.3*sdtw(x0.reshape(B*NC,-1),g_.reshape(B*NC,-1)).mean()
                    else:
                        l=l+0.3*sum((1-D.rho(x0[:,:q+1].reshape(B,-1),g_[:,:q+1].reshape(B,-1))).mean()
                                    for q in range(1,NC))/(NC-1)
                        l=l+0.6*(1-D.rho(xf,gf)).mean()
                    if fl.get('C'): l=l+0.3*Fn.mse_loss(wv,lsw[j])
                    opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                    with torch.no_grad():
                        for pe,pn in zip(ema.parameters(),net.parameters()):
                            pe.mul_(EMA_DECAY).add_(pn,alpha=1-EMA_DECAY)
                sch.step()
            ema.eval();tem=EV[np.isin(S[EV],te)]
            XS=XIQ_A if fl.get('Dg') else XIQ0
            with torch.no_grad():
                for b in range(0,len(tem),16):
                    kk=tem[b:b+16];mm=np.zeros(N,bool);mm[kk]=True
                    ci=T_(Cn,mm);pi=T_(Pn,mm)
                    for store,src in ((OUT,kk),(DNR,DON[kk])):
                        xi=torch.tensor(XS[src],device=dev)
                        tok,_,_=ema.cond(xi,ci,pi);tku,_,_=ema.cond(torch.zeros_like(xi),ci,pi)
                        store[nm][kk]=to_wave(gen(ema,tok,tku,len(kk),nchan),P).cpu().numpy()
            del ema;torch.cuda.empty_cache();print(f'  fold{fi}  {nm} done',flush=True)
    SE=S[EV];per=lambda v:np.array([v[SE==u].mean() for u in subs])
    UNI=np.array([np.mean([zn(D.X[i,c].reshape(-1)) for c in range(4)],0) for i in EV])
    eq=per(np.array([score(UNI[k],G[i],FS)['breath_f1'] for k,i in enumerate(EV)]))
    print(f'\n  {len(EV)} windows, {len(subs)} subjects, {NFOLD} folds.  four-way average {eq.mean():.4f}\n')
    print(f'  {"":26s} {"F1":>8s} {"|corr|":>8s} {"donor":>8s} {"real":>8s} {">0.5":>7s} {">0.6":>7s} '
          f'{"vs base":>9s} {"p":>7s}')
    R={}
    for nm,_ in ARMS:
        fw=np.array([score(OUT[nm][i],G[i],FS)['breath_f1'] for i in EV])
        v=per(fw);dn=per(np.array([score(DNR[nm][i],G[i],FS)['breath_f1'] for i in EV]))
        cc=per(np.array([abs(np.corrcoef(zn(OUT[nm][i]),zn(G[i]))[0,1]) for i in EV]))
        R[nm]=(v,dn,cc)
        b=R[ARMS[0][0]]
        d=f'{v.mean()-b[0].mean():+9.4f} {ttest_rel(v,b[0]).pvalue:7.4f}' if nm!=ARMS[0][0] else ''
        print(f'  {nm:26s} {v.mean():8.4f} {cc.mean():8.4f} {dn.mean():8.4f} {v.mean()-dn.mean():+8.4f} '
              f'{100*np.mean(fw>0.5):6.1f}% {100*np.mean(fw>0.6):6.1f}% {d}')
    np.savez_compressed(f'{PRE}/_ideas_abcde{GTAG}_s{SEED}.npz',subs=subs,eval_idx=EV,eq=eq,seed=SEED,
                        arms=np.array([a[0] for a in ARMS],dtype=object),
                        **{f'f1_{i}':R[a[0]][0] for i,a in enumerate(ARMS)},
                        **{f'dn_{i}':R[a[0]][1] for i,a in enumerate(ARMS)},
                        **{f'cc_{i}':R[a[0]][2] for i,a in enumerate(ARMS)},
                        **{f'wv_{i}':OUT[a[0]][EV] for i,a in enumerate(ARMS)},
                        **{f'dw_{i}':DNR[a[0]][EV] for i,a in enumerate(ARMS)})
