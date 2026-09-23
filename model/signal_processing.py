"""RESTORED clean RF-Carer pipeline (recovered from transcript after scratchpad loss, 2026-07-10).
CORRECT carrier (RATIO=0.375, FC=8.748GHz) + dual-domain bg (temporal EMA + range DBSCAN) + migrating
influenced-bin track + PCA/motion-removal extraction.
Note the carrier: RATIO=0.375 -> FC=8.748 GHz. An earlier variant using FC=7.29e9 with dbscan_bg and a
fixed bin produced a blurry Matrix Z and must not be used.
Reusable module: import signal_processing as RC.
"""
import os, glob, struct, numpy as np
from scipy.signal import butter, filtfilt, hilbert, decimate, resample, savgol_filter, correlate
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
import scipy.io as sio

import os as _os
# The carrier. The raw data's spectrum peaks at 7.0-7.2 GHz on every subject and both radars, which is
# Novelda's low band (7.29 GHz) pulled down a little by propagation - not the 8.748 GHz this file used
# to assume. The earlier note calling 7.29 'blurry' compared three changes at once, and blurriness is a
# property of |Z|, which a unit-modulus downconversion factor cannot affect at all.
# Correcting it leaves the phase swing unchanged (the residual is proportional to range, as the true
# phase is) but does change the candidates: oracle mix +0.0331 (p=0.027, 8/10 subjects).
# RFCARER_RATIO=0.375 restores the old behaviour for comparison.
FS=17.0; RATIO=float(_os.environ.get('RFCARER_RATIO','0.3125')); DEC=8
FC=RATIO*23.328e9; C=2.998e8; RES=0.0514; RLO,RHI=0.1,0.6
_DROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data")
_BATCHES=[os.path.join(_DROOT,"RF_Rawdata"),os.path.join(_DROOT,"RF_Rawdata2"),os.path.join(_DROOT,"RF_Rawdata3"),os.path.join(_DROOT,"RF_Rawdata4")]
RD=_BATCHES[0]  # legacy
def _dir(sub,kind):
    """resolve com/tv/biopac dir across all batches + case variants. kind in {com,tv,biopac}."""
    variants={'com':['com','Com'],'tv':['tv','Tv'],'biopac':['biopac','BIOPAC']}[kind]
    for base in _BATCHES:
        for v in variants:
            p=os.path.join(base,str(sub),v)
            if os.path.isdir(p): return p
    raise FileNotFoundError(f"{sub}/{kind}")
def _dat(sub,radar):
    base=_dir(sub,radar)
    c=glob.glob(f"{base}/**/xethru_datafloat_*.dat",recursive=True)  # batch3 nests + meta files
    if not c: c=[p for p in glob.glob(f"{base}/**/*.dat",recursive=True) if 'meta' not in os.path.basename(p)]
    c.sort(key=lambda p:len(p.split(os.sep)))  # shallowest (avoid nested duplicate)
    return c[0]
def _bio(sub): return glob.glob(f"{_dir(sub,'biopac')}/*.mat")[0]
SUBS_B1=["9","10","35","39","43"]
SUBS_B2=["1","6","25","28","41","42","47"]
SUBS_B3=["8","12","13","14","18","20","21","22","30","32","33","34","36","37","46","48","51","52","55","56","57","59","60"]
SUBS_B4=["2","7","15","17","19","23","29","44","53","58"]   # RF_Rawdata4 (the 10 missing from Spikformer 45)
SUBS_ALL=SUBS_B1+SUBS_B2+SUBS_B3+SUBS_B4   # 45 subjects (Spikformer parity, sub-orig-38 corrupt excluded)

def bpf(x):
    b,a=butter(3,[RLO/(FS/2),RHI/(FS/2)],'band'); return filtfilt(b,a,x)
def zn(x): return (x-x.mean())/(x.std()+1e-9)
def _sg(x,w,o=2):                                 # savgol with window clamped to a valid odd length<=len (short-record guard)
    x=np.asarray(x,float);n=len(x);w=int(w)|1
    if n<=o+2: return x
    if w>n: w=n if n%2 else n-1
    return savgol_filter(x,w,o) if w>o else x
def parse(p):
    d=open(p,'rb').read();off=0;Ln=len(d);fr=[]
    while off+12<=Ln:
        _,_,nc=struct.unpack_from('<III',d,off);off+=12
        if off+nc*4>Ln:break
        fr.append(np.frombuffer(d,'<f4',nc,off));off+=nc*4
    nb=max(set(len(x) for x in fr),key=[len(x) for x in fr].count)
    return np.array([x for x in fr if len(x)==nb],float)

def to_bb(rf):                                   # CORRECT downconversion: RATIO=0.375 in exponent
    n=np.arange(rf.shape[1])
    bb=hilbert(rf,axis=1)*np.exp(-1j*2*np.pi*RATIO*n)[None,:]
    return decimate(bb,DEC,axis=1,ftype='fir')
def sig_comp(W):                                 # Matrix X: Eq2 (d^2 range compensation), FC=8.748GHz
    d=np.arange(1,W.shape[1]+1)*RES
    return W*(d**2*np.exp(-1j*2*np.pi*FC*d/C))[None,:]
LAM=float(_os.environ.get('RFCARER_LAM','0.9'))   # EMA forgetting factor; 0.9 = authors' value (their 50 fps); at 17 fps its high-pass corner sits at ~0.30 Hz = inside the breathing band
def ema_bg(Xm,lam=None):                          # Matrix Y: temporal bg (EMA), Eq3-5
    lam=LAM if lam is None else lam
    T,B=Xm.shape;Bg=np.zeros_like(Xm);Bg[0]=Xm[0]
    for t in range(1,T): Bg[t]=lam*Bg[t-1]+(1-lam)*Xm[t]
    return Xm-Bg
RANGEBG=_os.environ.get('RFCARER_RANGEBG','bins')   # 'bins' = the implementation used so far (cluster ACROSS bins, one global mu);
                                                     # 'perbin' = the paper's Eq.6 literally (per bin i, DBSCAN over its T samples of |Y| and angle(Y), mu_A/mu_Phi of the dominant cluster)
def range_bg(Ym):                                 # Matrix Z: range-space bg (DBSCAN dominant cluster), Eq6
    if RANGEBG=='perbin': return range_bg_perbin(Ym)
    A=np.abs(Ym).mean(0);Ph=np.angle(Ym).mean(0)
    feat=np.stack([A/(A.max()+1e-9),np.cos(Ph),np.sin(Ph)],1)
    lab=DBSCAN(eps=0.15,min_samples=3).fit_predict(feat)
    mask=lab==np.bincount(lab[lab>=0]).argmax() if (lab>=0).any() else np.ones(len(A),bool)
    return (np.abs(Ym)-A[mask].mean())/(np.abs(Ym)+1e-9)*np.exp(1j*Ph[mask].mean())*Ym
def range_bg_perbin(Ym,eps=0.15,min_samples=3,step=1):
    """Eq.6 as written: for each range bin i, cluster the time samples (|Y(:,i)|, angle Y(:,i)) with DBSCAN, take the
    dominant cluster as the background of THAT bin, and subtract its mean amplitude / rotate by its mean phase."""
    T,B=Ym.shape;Z=np.empty_like(Ym);Aall=np.abs(Ym);amax=Aall.max()+1e-9
    for i in range(B):
        a=Aall[:,i];ph=np.angle(Ym[:,i])
        feat=np.stack([a/amax,np.cos(ph),np.sin(ph)],1)[::step]
        lab=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(feat)
        if (lab>=0).any():
            m=lab==np.bincount(lab[lab>=0]).argmax();sel=np.arange(0,T,step)[m]
        else: sel=np.arange(T)
        muA=a[sel].mean();muP=np.angle(np.exp(1j*ph[sel]).mean())
        Z[:,i]=(a-muA)/(a+1e-9)*np.exp(1j*muP)*Ym[:,i]
    return Z
def bbE(Ym,win=int(10*FS)):                       # breathing-band energy map
    T,B=Ym.shape;win=max(1,min(win,T));E=np.zeros((T,B));k=np.ones(win)/win   # win<=T (short-record guard)
    for i in range(B): E[:,i]=np.convolve(bpf(np.real(Ym[:,i]-Ym[:,i].mean()))**2,k,mode='same')
    return E
def track(A):                                     # migrating influenced-bin track (follows the body)
    T,B=A.shape;m=A/(A.max()+1e-12);st=int(np.argmax(m[:int(10*FS)].mean(0)));path=np.zeros(T,int);path[0]=st
    for t in range(T-1):
        it=path[t];lo=max(0,it-4);hi=min(B,it+5);ii=np.arange(lo,hi)
        cost=5*np.abs(m[t,it]-m[t+1,ii])+0.5*np.abs(it-ii)+0.01/(m[t,it]+m[t+1,ii]+1e-9)
        path[t+1]=ii[np.argmin(cost)]
    return np.clip(_sg(path,11).astype(int),0,B-1)
def body_track(Yabs, blo=None):
    """Track the MOVING body on temporal-bg amplitude |Y| (the chevron), local peak + continuity.
    Use this for LOS (body) — track() on breathing-energy locks onto the strong wall instead."""
    if blo is None: blo=int(0.6/RES)
    T,B=Yabs.shape; m=Yabs.copy(); m[:,:blo]=0
    ip=blo+int(np.argmax(m[:int(5*FS),blo:].mean(0))); p=np.zeros(T,int); p[0]=ip
    for t in range(1,T):
        lo=max(blo,ip-5); hi=min(B,ip+6); ip=lo+int(np.argmax(m[t,lo:hi])); p[t]=ip
    return np.clip(_sg(p,int(1*FS)).astype(int),0,B-1)
def body_track_v2(Yabs, blo=None, search=12, sg=7):
    """FIXED LOS tracker: remove persistent (wall) band via per-bin temporal median, then DP-track the
    MOVING body. Robust to wall lock-on. sg=savgol window (odd, tunable); search=transition band (bins)."""
    if blo is None: blo=int(0.6/RES)
    A=Yabs.astype(float).copy()
    A=A-np.median(A,axis=0,keepdims=True); A[A<0]=0; A[:,:blo]=0   # kill static wall band
    A=A/(A.max(1,keepdims=True)+1e-9)                              # per-frame normalize
    T,B=A.shape
    # banded Viterbi: maximize energy - continuity penalty
    NEG=-1e9; score=np.full(B,NEG); score[blo:]=A[0,blo:]; back=np.zeros((T,B),int)
    pen=0.15
    for t in range(1,T):
        ns=np.full(B,NEG)
        for b in range(blo,B):
            lo=max(blo,b-search);hi=min(B,b+search+1)
            cand=score[lo:hi]-pen*np.abs(np.arange(lo,hi)-b)
            j=lo+int(np.argmax(cand)); ns[b]=cand.argmax()*0+cand[np.argmax(cand)]+A[t,b]; back[t,b]=j
        score=ns
    p=np.zeros(T,int); p[-1]=int(np.argmax(score))
    for t in range(T-1,0,-1): p[t-1]=back[t,p[t]]
    sg=max(3,sg|1)
    return np.clip(_sg(p,sg).astype(int),0,B-1)
def ghost_bin(E):
    """fixed far-range wall/multipath bin = max mean breathing energy beyond 2.5 m (Ghost candidate)."""
    f0=int(2.5/RES); p=E.mean(0).copy(); p[:f0]=0; return int(np.argmax(p))
# --- STEP 1 (2026-07-14): body-trajectory reframe. No fixed LOS; body's range each moment IS the LOS.
#     Discriminator: body MIGRATES (transient energy per bin = high temporal CV); wall FIXED (low CV).
def mobility_cv(A, sm=int(3*FS)):
    """per-bin temporal coefficient of variation of smoothed |Y| -> high=transient(body), low=const(wall)."""
    k=np.ones(sm)/sm; As=np.apply_along_axis(lambda v:np.convolve(v,k,mode='same'),0,A)
    cv=As.std(0)/(As.mean(0)+1e-9); return cv/(cv.max()+1e-9)
def body_traj(A, blo=None, pen=0.35, search=8, sg=15):
    """DP-track migrating body: reward = per-frame-norm |Y| * mobility(CV); NO range cap; strong continuity.
    Returns (path, cv). Static wall suppressed by low CV weight. A = np.abs(Y)."""
    if blo is None: blo=int(0.6/RES)
    T,B=A.shape; cv=mobility_cv(A); m=A/(A.max(1,keepdims=True)+1e-9)
    R=m*cv[None,:]; R[:,:blo]=0
    NEG=-1e9; score=np.full(B,NEG); score[blo:]=R[0,blo:]; back=np.zeros((T,B),int)
    for t in range(1,T):
        ns=np.full(B,NEG)
        for b in range(blo,B):
            lo=max(blo,b-search); hi=min(B,b+search+1)
            cand=score[lo:hi]-pen*np.abs(np.arange(lo,hi)-b); j=int(np.argmax(cand))
            ns[b]=cand[j]+R[t,b]; back[t,b]=lo+j
        score=ns
    p=np.zeros(T,int); p[-1]=int(np.argmax(score))
    for t in range(T-1,0,-1): p[t-1]=back[t,p[t]]
    return np.clip(_sg(p,sg).astype(int),0,B-1), cv
def ghost_static(A, traj, blo=None):
    """REDEFINED Ghost = strongest STATIC reflector off the body trajectory (high mean, low CV, low occupancy).
    NOT the old '>2.5m argmax'. Returns (bin, static_score). A = np.abs(Y)."""
    if blo is None: blo=int(0.6/RES)
    mean_e=A.mean(0); mean_n=mean_e/(mean_e.max()+1e-9); cv=mobility_cv(A)
    occ=np.bincount(traj,minlength=len(mean_e)).astype(float); occ=occ/(occ.max()+1e-9)
    static=mean_n*(1-cv)*(1-occ); static[:blo]=-1e18   # forbid near band (argmax never picks <blo)
    return int(np.argmax(static)), static
def traj_speed(traj, sm=int(1*FS)):
    """|d(range)/dt| in m/s from a bin trajectory -> motion-gating signal (low = near-stationary)."""
    k=np.ones(sm)/sm; ts=np.convolve(traj.astype(float),k,mode='same')
    v=np.abs(np.gradient(ts))*RES*FS; return v
def kalman_rts(z, q=0.02, r=4.0):
    """constant-velocity Kalman forward + RTS backward smoother on a 1D bin path (STEP 2A stabilization).
    jerk ~14x lower than raw DP while preserving path length (real motion). Returns smoothed float path."""
    n=len(z);F=np.array([[1,1.],[0,1]]);H=np.array([[1,0.]])
    Q=q*np.array([[1/3,1/2],[1/2,1.]]);Rm=np.array([[r]])
    xf=np.zeros((n,2));Pf=np.zeros((n,2,2));xp=np.zeros((n,2));Pp=np.zeros((n,2,2))
    x=np.array([z[0],0.]);P=np.eye(2)*10
    for t in range(n):
        if t>0: x=F@x;P=F@P@F.T+Q
        xp[t]=x;Pp[t]=P
        y=z[t]-H@x;S=H@P@H.T+Rm;K=(P@H.T)@np.linalg.inv(S)
        x=x+(K@y).ravel();P=(np.eye(2)-K@H)@P;xf[t]=x;Pf[t]=P
    xs=xf.copy()
    for t in range(n-2,-1,-1):
        C=Pf[t]@F.T@np.linalg.inv(Pp[t+1]);xs[t]=xf[t]+C@(xs[t+1]-xp[t+1])
    return xs[:,0]
def body_traj_stab(A, blo=None, **kw):
    """STABILIZED body trajectory: near-raw DP -> Kalman/RTS -> integer bins. Returns (path, cv). STEP 2."""
    raw,cv=body_traj(A,blo=blo,sg=3,**kw)
    return np.clip(np.round(kalman_rts(raw.astype(float))).astype(int),0,A.shape[1]-1), cv
def ghost_static_free(A, blo=None):
    """Ghost/wall WITHOUT needing a trajectory: strongest STATIC reflector (high mean energy, low temporal CV).
    Used by the ghost-first pipeline, where the wall must be found BEFORE the body is tracked."""
    if blo is None: blo=int(0.6/RES)
    mean_n=A.mean(0)/(A.mean(0).max()+1e-9); cv=mobility_cv(A)
    st=mean_n*(1-cv); st[:blo]=-1e18
    return int(np.argmax(st)), st
def body_traj_ghost_first(A, margin_m=0.30, max_m=3.0, blo=None):
    """GHOST-FIRST body tracking. Extract the wall/Ghost first, delete that region (and everything beyond)
    from the map, then track the body in what is left. Without this the DP locks onto the static wall at the
    far end of each sweep, where the body echo is weakest (was happening for 40/45 subjects).
    The LOS band is additionally capped at max_m (default 3 m) so the body region matches the <3 m / >=3 m
    body-vs-ghost split used by the column normalisation; anything at or beyond that is Ghost by definition.
    Returns (traj, ghost_bin, cut_bin)."""
    if blo is None: blo=int(0.6/RES)
    gb,_=ghost_static_free(A,blo=blo)
    cut=min(gb-int(round(margin_m/RES)), int(max_m/RES))       # LOS strictly below max_m AND below the wall
    cut=max(blo+4, cut)
    Am=A.copy(); Am[:,cut:]=0.0                       # wall and beyond = Ghost region, not body
    tr,_=body_traj_stab(Am, blo=blo)
    return np.clip(tr,0,cut-1), gb, cut
def eq8_score(Y,bc,s0,wl):
    """paper Eq8: Psi(l_c)=max over resp band of product of |FFT| of 3 adjacent bins (Re & Im)."""
    seg=Y[s0:s0+wl]; T=len(seg); B=Y.shape[1]
    f=FS*np.arange(T//2+1)/T
    ra=int(np.argmax(f>=RLO)); rb=len(f)-1-int(np.argmax(f[::-1]<=RHI))
    if T<8 or rb<=ra or rb+1>T: return 0.0
    rp=np.ones(T); ip=np.ones(T); peo=np.zeros(T)
    for b in (bc-1,bc,bc+1):
        if b<0 or b>=B: continue
        s=seg[:,b]; rf=np.abs(np.fft.fft(np.real(s))/T); iff=np.abs(np.fft.fft(np.imag(s))/T)
        rp=rp*rf; ip=ip*iff; peo=peo+rp*2+ip*2
    band=peo[ra:rb+1]; return float(band.max()) if band.size else 0.0
def dynamic_select_path(Y,E,sub_sec=5):
    """paper 4.2.2: per sub-window, dynamically select LOS (body) vs Ghost (wall) by Eq8.
    Returns per-frame influenced-bin path (piecewise-constant per sub-window) + selection flags."""
    B=Y.shape[1]; T=len(Y); sw=int(sub_sec*FS)
    bp=body_track(np.abs(Y)); gb=ghost_bin(E)
    path=np.zeros(T,int); sel=np.zeros(T,int)   # sel: 0=LOS, 1=Ghost
    for s0 in range(0,T,sw):
        w=min(sw,T-s0); los=int(np.median(bp[s0:s0+w]))
        pick_gho = eq8_score(Y,gb,s0,w) > eq8_score(Y,los,s0,w)
        path[s0:s0+w]=gb if pick_gho else los; sel[s0:s0+w]=1 if pick_gho else 0
    return path,sel,gb
def pca1(z):
    zc=z-z.mean();M=np.vstack([np.real(zc),np.imag(zc)]);M=M-M.mean(1,keepdims=True)
    u,s,vt=np.linalg.svd(M,full_matrices=False);return vt[0]*s[0]
def rout(sig,ref):                                # motion removal: regress out amplitude (bulk motion)
    M=(bpf(ref)-bpf(ref).mean())[:,None];return sig-M@np.linalg.lstsq(M,sig,rcond=None)[0]
def domf(x):
    X=np.abs(np.fft.rfft(zn(x)*np.hanning(len(x)),n=8192));f=np.fft.rfftfreq(8192,1/FS)
    b=(f>=RLO)&(f<=RHI);return float(f[b][np.argmax(X[b])]*60)
def csL(a,b):
    a=zn(a);b=zn(b);n=min(len(a),len(b));a,b=a[:n],b[:n]
    xc=correlate(a,b,'full');lg=np.arange(-n+1,n);sel=np.abs(lg)<=int(2*FS);L=lg[sel][np.argmax(np.abs(xc[sel]))]
    aa,bb=(a[L:],b[:n-L]) if L>=0 else (a[:n+L],b[-L:]);return abs(pearsonr(aa,bb)[0])
import re
def _mmss(n):
    m=re.search(r"(\d{2})_(\d{2})_(\d{2})\.mat",n) or re.search(r"_(\d{2})(\d{2})(\d{2})\.dat",n)
    return int(m.group(2))*60+int(m.group(3))
def offset(sub,radar='com'):
    b=os.path.basename(_bio(sub));c=os.path.basename(_dat(sub,radar))
    o=(_mmss(c)-_mmss(b))%3600; return o-3600 if o>1800 else o
def gt_load(sub):
    m=sio.loadmat(_bio(sub));r=m['data'][:,0].astype(float)   # col0 = RSP (both batches)
    return bpf(resample(r,int(round(len(r)/250*FS))))

def full(sub,radar='com'):
    """raw -> W,X,Y,Z + breathing-energy E + global track path. Returns dict."""
    rf=parse(_dat(sub,radar))
    W=to_bb(rf); X=sig_comp(W); Xc=X-X.mean(0,keepdims=True); Y=ema_bg(Xc); Z=range_bg(Y); E=bbE(Y)
    return dict(W=W,X=X,Y=Y,Z=Z,E=E,path=track(E))
def extract_window(Y,E,s0,wl,gw):
    """clean extraction: track bin in window -> PCA -> motion-removal -> bpf."""
    pth=track(E[s0:s0+wl]); idx=np.arange(s0,s0+wl)
    return bpf(rout(pca1(Y[idx,pth]),np.abs(Y[idx,pth])))
def extract_hz(Z,binseq,idx):
    """DEFAULT hybrid extraction: signal from Matrix Z at given bins (track computed on Y elsewhere)."""
    sig=Z[idx,binseq] if np.ndim(binseq) else Z[idx,binseq]
    return bpf(rout(pca1(sig),np.abs(sig)))
def concat_extract(Z,binpath,s0,wl,chunk_s=3):
    """extract per short chunk (local PCA axis) and concatenate with continuity sign-align -> long seq.
    binpath = per-frame influenced-bin (tracked on Y). Returns bandpassed concatenated signal length wl."""
    cw=int(chunk_s*FS); pieces=[]; prev_end=None
    for c0 in range(s0,s0+wl,cw):
        c1=min(c0+cw,s0+wl); ci=np.arange(c0,c1)
        sig=Z[ci,binpath[ci]] if np.ndim(binpath) else Z[ci,binpath]
        r=rout(pca1(sig),np.abs(sig))                     # local displacement axis (pre-bpf)
        if prev_end is not None and len(r) and r[0]*prev_end<0: r=-r   # continuity sign-align (GT-free)
        if len(r): prev_end=r[-1]
        pieces.append(r)
    return bpf(np.concatenate(pieces))

if __name__=='__main__':
    # verify restoration: best-window CS per subject (should recover ~0.62 for COM)
    for sub in ["9","10","35","39","43"]:
        d=full(sub,'com'); g=gt_load(sub); off=int(round(offset(sub)*FS)); Y,E=d['Y'],d['E']
        wl=int(30*FS); best=-1
        for s0 in range(0,len(Y)-wl,int(15*FS)):
            gi=s0+off
            if gi<0 or gi+wl>len(g): continue
            gw=g[gi:gi+wl]
            if gw.std()<1e-9: continue
            try: c=csL(extract_window(Y,E,s0,wl,gw),gw)
            except Exception: continue
            best=max(best,c)
        print(f"sub{sub} COM best-window CS(30s) = {best:.3f}")
