"""Conditional diffusion transformer that REGENERATES the 42 s breathing waveform.

This is the model that was asked for and that the earlier scripts only carried the name of. What was there
before was a CNN encoder, a GRU and a two-layer MLP decoder emitting each 3 s chunk in isolation - no
diffusion, no transformer, no continuity. It left 7.8% of its energy above the 0.6 Hz passband and a step at
every chunk seam 6.2x the size of a normal sample step, against 0.0% and 1.04x for the GT.

Structure
  tokens        14 tokens, one per 3 s chunk, so the stitching the user described happens inside attention
                rather than by concatenation afterwards
  conditioning  per chunk: the four candidates, a learned softmax SELECTION RATIO over them (kept explicit
                so it can be read off and plotted), and the 12 geometry features (r_com, r_tv, their gap,
                |gap|, tanh(gap/0.5), position fraction, corner distance, both speeds, approach state and
                both log Ghost/LOS ratios). Amplitude is deliberately absent - sig_comp already applies a
                d^2 range compensation, and what is left of amplitude predicts the best candidate in 27% of
                chunks against a 25% chance level.
  denoiser      DiT blocks: self-attention across the 14 chunks with adaLN-Zero modulation from the
                diffusion timestep and the conditioning summary. Attention across chunks is what lets the
                model fix a seam by changing BOTH sides of it, which a causal decoder cannot do.
  localisation  a 14-class head on the RANGE PROFILE, on its own path. Position is where the energy sits in
                range, which is an image, not something to infer from the shape of an extracted breathing
                trace. Measured per chunk, subject-wise: the profile image gives 0.924 and the breathing
                waveforms 0.214, and the head used to be given the waveforms, which is why it scored 0.348.
                The path is kept SEPARATE from the candidate encoder because the waveform interferes -
                geometry alone reaches 0.891 and geometry plus waveform only 0.809. Its output still feeds
                the conditioning summary, so the same model produces it, as asked.

Motion noise. The dataset chunks the subject's walking every 3 s, so transitions inject spikes. The step-1
conditioner (robust z against a local median, runs shorter than 1.5 s repaired by interpolation, runs longer
released because a breath at 0.6 Hz lasts at least 1.67 s) was never applied in the geometry dataset. It is
applied here to every candidate before it is seen.

Losses, keeping the 3 s local / 42 s global format
    L = MSE(eps, eps_hat)                                   the diffusion objective
      + CE(localisation)
      + mean over k of [1 - rho(x0_hat[:k], gt[:k])]        LOCAL: every accumulated prefix, 3 s .. 42 s
      + 2 * [1 - rho(x0_hat, gt)]                           GLOBAL: the whole 42 s, weighted twice
rho is SIGNED, not |rho|. The absolute value hands the model a free sign flip per window, and a correlation
loss that can flip signs is one more way to win without matching anything.

Sampling is DDIM, and the sample is projected onto 0.1-0.6 Hz at every step - the same passband the inputs
and the GT both live in, so it is a constraint the data already obeys rather than a smoothing knob.

Controls, all reported: a phase-randomised surrogate of each reconstruction, the same model run with a
DIFFERENT window's conditioning (shuffled geometry), a constant-rate oscillator at the cohort median rate,
and the plain 1/4 candidate mix. Five-fold subject-wise CV.

Scoring is on DISJOINT windows only. The dataset is built with a 6 s hop so neighbouring windows share 36
of their 42 s; averaging over all of them counts the same stretch about seven times and lets one good
stretch stand in for the subject. Training still uses every window - overlap is extra data there - but the
reported numbers come from every seventh window, which is the first one that does not overlap its
predecessor.
Saves preprocessed/_dit_out.npz and prints the panel.
"""
import _path  # noqa: F401  (see _path.py)
import os,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
import signal_processing as RC
from scipy.stats import pearsonr,ttest_rel
from conditioner import condition
from metrics import score,surrogate_control
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed'
dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0);np.random.seed(0)
# GEO selects which dataset to run on. The conditioned-candidate cache has to follow it, or a run on the
# re-synced radars would silently reuse candidates repaired from the old, mis-timed ones.
GEO=os.environ.get('GEO','_geo_dataset.npz');GTAG=GEO.replace('_geo_dataset','').replace('.npz','')
d=np.load(f'{PRE}/{GEO}',allow_pickle=True)
X0=d['X'];C=d['C'];G=d['G'];ZONE=d['ZONE'];S=d['S'];CTX=list(d['ctx_names'])
PROF=d['PROF']
NC=int(d['NC']);CH=int(d['CH']);FS=float(d['FS']);W=NC*CH;NCTX=C.shape[-1]
HOP_S=float(d['HOP'])/FS if 'HOP' in d.files else 6.0   # row spacing in seconds (6 s grid unless an off-grid set)
CAND=['COM-LOS','COM-Ghost','TV-LOS','TV-Ghost'];GAPI=CTX.index('gap')
T_DIFF=1000;DDIM=50;DM=192;NBLK=4;NHEAD=4;EP=int(os.environ.get('EP','60'))
# CALSTATS=1: the AdaLN summary also receives a per-sample calibration-statistics vector c_u
# (amplitude-CV and interval-CV of an 84 s belt block, z-scored) through cu_mlp = Linear(2,dm)-SiLU-Linear(dm,dm) with the
# last layer zero-initialised. Unset (default): no module is created and forward() is the pre-existing code path.
CALSTATS=os.environ.get('CALSTATS','0')=='1';CU_DIM=2
print(f'{len(X0)} windows, {len(np.unique(S))} subjects, dev={dev}, epochs={EP}, data={GEO}')
# ---- motion-noise repair on every candidate, missing until now -------------------------------------
cf=f'{PRE}/_dit_X_conditioned{GTAG}.npy'
if os.path.exists(cf):
    X=np.load(cf)
else:
    X=np.empty_like(X0);rep=np.zeros((len(X0),4),np.float32)
    for i in range(len(X0)):
        for c in range(4):
            y,f,fr=condition(X0[i,c].reshape(-1));X[i,c]=y.reshape(NC,CH);rep[i,c]=fr
        if i%200==0: print(f'  conditioning {i}/{len(X0)}',end='\r')
    np.save(cf,X);print(f'  motion-noise repair: {rep.mean():.2%} of samples repaired'+' '*20)
zn=lambda x:(x-x.mean())/(x.std()+1e-9)
X=np.stack([np.stack([zn(X[i,c].reshape(-1)).reshape(NC,CH) for c in range(4)]) for i in range(len(X))])
cm=C.reshape(-1,NCTX).mean(0);cs=C.reshape(-1,NCTX).std(0)+1e-6
Cn=((C-cm)/cs).astype(np.float32)
NPROF=PROF.shape[-1]
pm_=PROF.reshape(-1,NPROF).mean(0);ps_=PROF.reshape(-1,NPROF).std(0)+1e-6
Pn=((PROF-pm_)/ps_).astype(np.float32)
def fold_norm(trm):
    """Normalise the geometry features and the range profile using ONLY the training subjects' statistics.

    Cn and Pn above are fitted on every window, test subjects included. That is a leak, and not a small one
    here: refitting on the training subjects alone moves a held-out window's normalised range profile by up
    to 3.0 standard deviations. Cn/Pn are left in place so older scripts still run, but anything doing
    subject-wise cross-validation must call this inside the fold loop and use what it returns.
    """
    cm_=C[trm].reshape(-1,NCTX).mean(0);cs_=C[trm].reshape(-1,NCTX).std(0)+1e-6
    pm2=PROF[trm].reshape(-1,NPROF).mean(0);ps2=PROF[trm].reshape(-1,NPROF).std(0)+1e-6
    return ((C-cm_)/cs_).astype(np.float32),((PROF-pm2)/ps2).astype(np.float32)
Gt=G.reshape(len(G),NC,CH).astype(np.float32)
# ---- band projection: the passband the data already obeys -----------------------------------------
fr=torch.fft.rfftfreq(W,1/FS).to(dev);BAND=((fr>=RC.RLO)&(fr<=RC.RHI)).float()
def band(x):
    """x (B,NC,CH) -> same, with everything outside 0.1-0.6 Hz removed"""
    B=x.shape[0];v=x.reshape(B,-1)
    return torch.fft.irfft(torch.fft.rfft(v)*BAND,n=W).reshape(B,NC,CH)
# ---- diffusion schedule ----------------------------------------------------------------------------
bt=torch.linspace(1e-4,0.02,T_DIFF,device=dev);al=1-bt;ab=torch.cumprod(al,0)
def temb(t,dim=DM):
    h=dim//2;f=torch.exp(-math.log(10000)*torch.arange(h,device=t.device)/h)
    a=t.float()[:,None]*f[None]
    return torch.cat([a.sin(),a.cos()],-1)
class Blk(nn.Module):
    def __init__(s,dm=DM,nh=NHEAD):
        super().__init__()
        s.n1=nn.LayerNorm(dm,elementwise_affine=False);s.att=nn.MultiheadAttention(dm,nh,batch_first=True)
        s.n2=nn.LayerNorm(dm,elementwise_affine=False)
        s.mlp=nn.Sequential(nn.Linear(dm,dm*4),nn.GELU(),nn.Linear(dm*4,dm))
        s.ada=nn.Sequential(nn.SiLU(),nn.Linear(dm,6*dm));nn.init.zeros_(s.ada[1].weight);nn.init.zeros_(s.ada[1].bias)
    def forward(s,x,c):
        p=s.ada(c).chunk(6,-1);sh1,sc1,g1,sh2,sc2,g2=[q.unsqueeze(1) for q in p]
        h=s.n1(x)*(1+sc1)+sh1;a,_=s.att(h,h,h);x=x+g1*a
        h=s.n2(x)*(1+sc2)+sh2;x=x+g2*s.mlp(h)
        return x
class DiT(nn.Module):
    def __init__(s,dm=DM):
        super().__init__()
        s.cenc=nn.Sequential(nn.Conv1d(1,32,7,padding=3),nn.GELU(),nn.Conv1d(32,64,5,padding=2),nn.GELU(),
                             nn.Flatten(),nn.Linear(64*CH,dm))
        s.gctx=nn.Sequential(nn.Linear(NCTX,64),nn.GELU(),nn.Linear(64,dm))
        # localisation reads the range profile on its own path - the waveform interferes with it
        s.penc=nn.Sequential(nn.Linear(NPROF,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        s.selw=nn.Sequential(nn.Linear(2*dm,64),nn.GELU(),nn.Linear(64,1))     # selection ratio, explicit
        s.loc=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,14))
        s.xin=nn.Linear(CH,dm);s.pos=nn.Parameter(torch.randn(1,NC,dm)*.02)
        s.cproj=nn.Sequential(nn.Linear(2*dm+14,dm),nn.GELU(),nn.Linear(dm,dm))
        s.blocks=nn.ModuleList([Blk() for _ in range(NBLK)])
        s.nout=nn.LayerNorm(dm,elementwise_affine=False)
        s.out=nn.Linear(dm,CH);nn.init.zeros_(s.out.weight);nn.init.zeros_(s.out.bias)
        s.tmlp=nn.Sequential(nn.Linear(dm,dm),nn.SiLU(),nn.Linear(dm,dm))
        if CALSTATS:
            s.cu_mlp=nn.Sequential(nn.Linear(CU_DIM,dm),nn.SiLU(),nn.Linear(dm,dm))
            nn.init.zeros_(s.cu_mlp[2].weight);nn.init.zeros_(s.cu_mlp[2].bias)
            # z-score statistics of the fold's TRAINING subjects, set by the trainer and saved with the weights
            s.register_buffer('cu_mu',torch.zeros(CU_DIM));s.register_buffer('cu_sd',torch.ones(CU_DIM))
    def cu_norm(s,raw):
        """raw (B,2) belt statistics -> z-scored with the fold's training statistics"""
        return (raw-s.cu_mu)/s.cu_sd
    def cond(s,x,c,pr):
        B=x.shape[0]
        e=s.cenc(x.reshape(B*4*NC,1,CH)).reshape(B,4,NC,-1).permute(0,2,1,3)    # (B,NC,4,dm)
        g=s.gctx(c)                                                             # (B,NC,dm)
        w=torch.softmax(s.selw(torch.cat([e,g.unsqueeze(2).expand(-1,-1,4,-1)],-1)).squeeze(-1),-1)
        f=(e*w.unsqueeze(-1)).sum(2)                                            # selected features
        lz=s.loc(s.penc(pr))                                                    # localisation from the profile
        tok=s.cproj(torch.cat([f,g,lz],-1))
        return tok,w,lz
    def forward(s,xt,t,tok,cu=None):
        h=s.xin(xt)+s.pos+tok
        c=s.tmlp(temb(t))+tok.mean(1)
        if cu is not None:                       # cu: (B,2) z-scored calibration statistics (CALSTATS=1 only)
            assert CALSTATS,'cu given but CALSTATS is not set'
            c=c+s.cu_mlp(cu)
        for b in s.blocks: h=b(h,c)
        return s.out(s.nout(h))
def rho(a,b):
    a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)+1e-8)
@torch.no_grad()
def sample(net,tok,n,steps=DDIM):
    x=torch.randn(n,NC,CH,device=dev)
    ts=torch.linspace(T_DIFF-1,0,steps).long().to(dev)
    for i in range(steps):
        t=ts[i].repeat(n);e=net(x,t,tok)
        a=ab[ts[i]];x0=(x-(1-a).sqrt()*e)/a.sqrt()
        x0=band(x0)                                    # the passband the data obeys, enforced each step
        if i<steps-1:
            a2=ab[ts[i+1]];x=a2.sqrt()*x0+(1-a2).sqrt()*e
        else: x=x0
    return x
subs=np.unique(S);pm=np.random.RandomState(0).permutation(len(subs))
folds=[subs[pm[i::5]] for i in range(5)]
# every seventh window of a subject is the first that does not overlap its predecessor (42 s / 6 s hop)
DISJ=np.zeros(len(X),bool)
for u in subs:
    q=np.flatnonzero(S==u);DISJ[q[::7]]=True
print(f'scoring on {DISJ.sum()} disjoint windows of {len(X)} '
      f'({DISJ.sum()/len(subs):.1f} per subject); training uses all of them')
PRED=np.zeros((len(X),NC,CH),np.float32);PRED_SH=np.zeros_like(PRED)
WSEL=np.zeros((len(X),NC,4),np.float32);LOCP=np.zeros((len(X),NC),np.int64)
HIST={'eps':[],'local':[],'global':[],'loc':[]}
# Other scripts import this module for its data, constants and the DiT class, then load their own
# checkpoints. Everything below is this file's OWN experiment - without the guard every such import
# paid for a full 5-fold retrain and a scoring pass. Nothing below is read by another script
# (checked: no importer references PRED, WSEL, LOCP, HIST, folds or DISJ).
if __name__=='__main__':
    for fi,te in enumerate(folds):
        trm=~np.isin(S,te);tem=np.isin(S,te)
        T_=lambda a,m:torch.tensor(a[m],device=dev)
        xt_,ct_,gt_,zt_,pt_=T_(X,trm),T_(Cn,trm),T_(Gt,trm),T_(ZONE,trm),T_(Pn,trm)
        xe,ce,pe_=T_(X,tem),T_(Cn,tem),T_(Pn,tem)
        ridx=np.random.RandomState(fi).permutation(int(tem.sum()))
        net=DiT().to(dev);opt=torch.optim.AdamW(net.parameters(),2e-4,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP);N=len(xt_);bs=32
        for ep in range(EP):
            net.train();idx=torch.randperm(N,device=dev);a1=[];a2=[];a3=[];a4=[]
            for b in range(0,N,bs):
                j=idx[b:b+bs];B=len(j)
                tok,w,lz=net.cond(xt_[j],ct_[j],pt_[j])
                t=torch.randint(0,T_DIFF,(B,),device=dev)
                g0=gt_[j];eps=torch.randn_like(g0)
                a=ab[t][:,None,None]
                xn_=a.sqrt()*g0+(1-a).sqrt()*eps
                ep_=net(xn_,t,tok)
                l_eps=Fn.mse_loss(ep_,eps)
                x0=(xn_-(1-a).sqrt()*ep_)/a.sqrt()
                ll=0.
                for k in range(1,NC):
                    ll=ll+(1-rho(x0[:,:k+1].reshape(B,-1),g0[:,:k+1].reshape(B,-1))).mean()
                ll=ll/(NC-1)
                lg=(1-rho(x0.reshape(B,-1),g0.reshape(B,-1))).mean()
                l_loc=Fn.cross_entropy(lz.reshape(-1,14),zt_[j].reshape(-1),ignore_index=-1)
                loss=l_eps+l_loc+0.3*ll+0.6*lg
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                a1.append(l_eps.item());a2.append(ll.item());a3.append(lg.item());a4.append(l_loc.item())
            sch.step()
            if fi==0:
                HIST['eps'].append(np.mean(a1));HIST['local'].append(np.mean(a2))
                HIST['global'].append(np.mean(a3));HIST['loc'].append(np.mean(a4))
            if ep%15==0: print(f'  fold{fi} ep{ep:3d} eps {np.mean(a1):.4f} local {np.mean(a2):.3f} '
                               f'global {np.mean(a3):.3f} loc {np.mean(a4):.3f}')
        net.eval()
        with torch.no_grad():
            tok,w,lz=net.cond(xe,ce,pe_)
            PRED[tem]=sample(net,tok,len(xe)).cpu().numpy()
            WSEL[tem]=w.cpu().numpy();LOCP[tem]=lz.argmax(-1).cpu().numpy()
            tok_sh,_,_=net.cond(xe,ce[torch.tensor(ridx,device=dev)],pe_)
            PRED_SH[tem]=sample(net,tok_sh,len(xe)).cpu().numpy()
        m=ZONE[tem]>=0
        print(f'  fold{fi} zone acc {(LOCP[tem][m]==ZONE[tem][m]).mean():.3f}')
    np.savez_compressed(f'{PRE}/_dit_out{GTAG}.npz',PRED=PRED,PRED_SH=PRED_SH,WSEL=WSEL,LOCP=LOCP,S=S,HIST=HIST)
    print('\nscoring ...')
    rec=PRED.reshape(len(X),-1);rsh=PRED_SH.reshape(len(X),-1)
    unif=np.stack([zn(X[i].mean(0).reshape(-1)) for i in range(len(X))])
    t_=np.arange(W)/FS
    f0=np.median([RC.domf(G[i])/60 for i in range(0,len(G),7)])
    osc=np.stack([zn(np.sin(2*np.pi*f0*t_+2*np.pi*np.random.RandomState(i).rand())) for i in range(len(X))])
    def oob(x):
        A=np.abs(np.fft.rfft(zn(x)*np.hanning(len(x)),4096))**2;f=np.fft.rfftfreq(4096,1/FS)
        return float(A[f>RC.RHI].sum()/(A[f>0.02].sum()+1e-12))
    def seam(x):
        d1=np.abs(np.diff(x));return float(np.mean([d1[k*CH-1] for k in range(1,NC)])/(np.median(d1)+1e-12))
    def panel(A,tag,su=True):
        # scored on DISJOINT windows only - the dataset has a 6 s hop, so neighbours share 36 of their 42 s and
        # averaging over all of them counts the same stretch about seven times
        f=[];s=[];o=[];sm=[]
        for i in np.flatnonzero(DISJ):
            f.append(score(A[i],G[i],FS)['breath_f1'])
            if su: s.append(surrogate_control(A[i],G[i],FS,n=4)['breath_f1'])
            o.append(oob(A[i]));sm.append(seam(A[i]))
        return tag,np.array(f),(np.array(s) if su else None),np.mean(o),np.mean(sm)
    R=[panel(rec,'DiT'),panel(rsh,'DiT, shuffled geometry'),panel(unif,'uniform 1/4 mix'),
       panel(osc,'constant-rate oscillator'),panel(G,'GT',su=False)]
    print(f'\n{"arm":26s} {"breath F1":>10s} {"surrogate":>10s} {"F1 - surr":>10s} {">0.6 Hz":>9s} {"seam":>8s}')
    for tag,f,s,o,sm in R:
        sv=s.mean() if s is not None else np.nan
        print(f'{tag:26s} {f.mean():10.4f} {sv:10.4f} {f.mean()-sv:+10.4f} {o:8.1%} {sm:7.2f}x')
    SD=S[DISJ]
    bysub=lambda v:np.array([v[SD==u].mean() for u in subs])
    a=bysub(R[0][1]);sh=bysub(R[1][1]);sr=bysub(R[0][2]);un=bysub(R[2][1]);oc=bysub(R[3][1])
    print(f'\npaired across {len(subs)} subjects (MDE about 0.0124):')
    print(f'  DiT vs its own surrogate      {a.mean()-sr.mean():+.4f}  p {ttest_rel(a,sr).pvalue:.4f}')
    print(f'  DiT vs shuffled geometry      {a.mean()-sh.mean():+.4f}  p {ttest_rel(a,sh).pvalue:.4f}')
    print(f'  DiT vs constant-rate oscill.  {a.mean()-oc.mean():+.4f}  p {ttest_rel(a,oc).pvalue:.4f}')
    print(f'  DiT vs uniform 1/4 mix        {a.mean()-un.mean():+.4f}  p {ttest_rel(a,un).pvalue:.4f}')
    m=(ZONE>=0)&DISJ[:,None];print(f'\nzone accuracy (disjoint windows) {(LOCP[m]==ZONE[m]).mean():.3f}')
    gap=C[DISJ][:,:,GAPI].reshape(-1);wf=WSEL[DISJ].reshape(-1,4)
    r=pearsonr(gap,wf[:,0]+wf[:,1])
    print(f'selection ratio vs geometry: corr(gap, COM pair weight) = {r.statistic:+.3f} p {r.pvalue:.1e}')
    for k in range(4):
        print(f'  {CAND[k]:11s} mean {wf[:,k].mean():.3f} sd {wf[:,k].std():.3f} '
              f'corr(gap) {pearsonr(gap,wf[:,k]).statistic:+.3f}')
