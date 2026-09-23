"""STEP 1 of the DiT pipeline — the chunk conditioner.

One job only: take a 4-6 s chunk of an extracted candidate and remove the transient spikes that appear when
the subject stops and then walks off, so the waveform going into the fusion stage is smooth. Nothing here
selects, combines or judges - those are steps 2-4.

Measured basis for the design (861 windows):
  - transients are 5.70% of samples, and 58% of them fall within 2 s of a movement onset
  - their height is median |robust z| = 6.6, so a threshold separates them from the surround
  - they carry 30.7% of a 42 s correlation, i.e. a few samples can decide a chunk's score

Design:
  detect   robust z against a local median/MAD computed on a WIDE context (not inside the chunk, where MAD
           would be estimated from the spike itself), THEN a duration test. Amplitude alone cannot separate a
           movement spike from a deep breath in an otherwise quiet stretch - the band is 0.1-0.6 Hz, so a
           breath lasts at least 1.7 s while a movement transient is around a second. Only runs SHORTER than
           MAXDUR are treated as transients; longer excursions are left alone. Without this, COM-LOS lost a
           2.12 s breath and corr(before,after) fell to 0.67.
  repair   flagged runs are replaced by cubic interpolation ACROSS the gap, not clipped - clipping leaves a
           corner, which is itself a discontinuity for the later alignment stage. A run that touches either
           end is held at the nearest valid value instead: a spline has nothing to interpolate between there
           and extrapolates away (this broke COM-LOS in the first version, corr(before,after) 0.65).
  re-band  the repaired signal is passed through the same 0.1-0.6 Hz filter so the repair cannot introduce
           out-of-band content, then rescaled by a single gain fitted on the UNFLAGGED samples - filtering an
           already-band-passed signal attenuates it, and the untouched part must keep its original scale.
  report   per chunk, the fraction of samples repaired. Step 3 can use this as a discard feature; here it is
           only reported.

Exports `condition(x, thr=4.0)` for later steps to import.
Saves MTR/dit_step1_conditioner.png
"""
import os,numpy as np
import signal_processing as RC
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed';MTR='/home/user1/Desktop/UWB_BIOPAC/MTR'
FS,RES=RC.FS,RC.RES
CTX=int(12*FS)          # context for the robust baseline
THR=4.0                 # |robust z| above this is a transient
GROW=int(0.15*FS)       # widen each flagged run by this much on each side
MAXDUR=int(1.5*FS)      # a run longer than this is in-band by duration (>=1.7 s = a breath)
zn=lambda x:(x-x.mean())/(x.std()+1e-9)
def flag(x,thr=THR,ctx=CTX):
    """GT-free transient mask: robust z against a local median/MAD over a wide context"""
    w=ctx|1
    m=medfilt(x,w);a=np.abs(x-m);mad=medfilt(a,w)+1e-9
    z=(x-m)/(1.4826*mad);f=np.abs(z)>thr
    if GROW>0 and f.any():                       # widen runs so the shoulders go too
        k=np.ones(2*GROW+1)
        f=np.convolve(f.astype(float),k,'same')>0
    if f.any():                                  # release runs that are too long to be a movement spike
        d=np.diff(np.concatenate([[0],f.astype(int),[0]]))
        for a,b in zip(np.flatnonzero(d>0),np.flatnonzero(d<0)):
            if b-a>MAXDUR: f[a:b]=False
    return f,z
def condition(x,thr=THR,ctx=CTX):
    """remove transient spikes and hand back a smooth in-band signal.
    Returns (cleaned, mask, fraction_repaired)."""
    x=np.asarray(x,float);f,_=flag(x,thr,ctx)
    if f.all() or not f.any(): return x.copy(),f,float(f.mean())
    i=np.arange(len(x));keep=~f
    y=x.copy();lo_i,hi_i=i[keep][0],i[keep][-1]
    inner=f&(i>lo_i)&(i<hi_i)                     # only these can be interpolated between valid samples
    if keep.sum()>=4 and inner.any():
        y[inner]=CubicSpline(i[keep],x[keep])(i[inner])
    elif inner.any():
        y[inner]=np.interp(i[inner],i[keep],x[keep])
    y[i<lo_i]=x[lo_i];y[i>hi_i]=x[hi_i]           # edge runs: hold, never extrapolate
    yb=RC.bpf(y-y.mean())
    # one gain fitted on the untouched samples, so conditioning cannot rescale the waveform
    d=float(np.sum(yb[keep]**2))
    if d>1e-12: yb*=float(np.sum(x[keep]*yb[keep]))/d
    return yb,f,float(f.mean())
if __name__=='__main__':
    SUBS=['sub32','sub08','sub42','sub28','sub20','sub02','sub52']
    S0=int(3*FS);SPAN=int(42*FS)
    CAND=['COM-LOS','COM-Ghost','TV-LOS','TV-Ghost'];PCOL=['#27ae60','#e67e22','#2980b9','#8e44ad']
    print(f'conditioner: |robust z|>{THR}, {CTX/FS:.0f} s context, runs widened {GROW/FS*1000:.0f} ms, '
          f'runs longer than {MAXDUR/FS:.1f} s released\n')
    print(f"{'sub':8s} {'candidate':12s} {'repaired':>9s} {'runs':>5s} {'longest':>8s} | "
          f"{'corr(bef,aft)':>13s} | {'|corr| GT bef':>13s} {'aft':>7s} {'Δ':>7s}")
    SUM=[]
    for NM in SUBS:
        z=np.load(f'{PRE}/{NM}/data.npz',allow_pickle=True)
        g=z['gt_aligned'];T=min(len(g),z['Z_com'].shape[0],z['Z_tv'].shape[0])
        if T<S0+SPAN: print(f'{NM}: too short');continue
        PADX=int(6*FS);lo=max(0,S0-PADX);hi=min(T,S0+SPAN+PADX);ii=np.arange(lo,hi);off=S0-lo
        F=[RC.extract_hz(z['Z_com'][:T],z['traj_com'].astype(int)[:T][ii],ii),
           RC.extract_hz(z['Z_com'][:T],np.full(len(ii),int(z['ghost_com'])),ii),
           RC.extract_hz(z['Z_tv'][:T],z['traj_tv'].astype(int)[:T][ii],ii),
           RC.extract_hz(z['Z_tv'][:T],np.full(len(ii),int(z['ghost_tv'])),ii)]
        C=[condition(x) for x in F]
        gw=g[S0:S0+SPAN];t=np.arange(SPAN)/FS+S0/FS
        for k,(y,f,fr) in enumerate(C):
            x=F[k];lens=[]
            if f.any():
                d=np.diff(np.concatenate([[0],f.astype(int),[0]]))
                lens=(np.flatnonzero(d<0)-np.flatnonzero(d>0))/FS
            a=abs(float(np.mean(zn(x[off:off+SPAN])*zn(gw))))
            b=abs(float(np.mean(zn(y[off:off+SPAN])*zn(gw))))
            SUM.append((NM,CAND[k],fr,b-a))
            print(f"{NM if k==0 else '':8s} {CAND[k]:12s} {fr:9.1%} {len(lens):5d} "
                  f"{max(lens) if len(lens) else 0:7.2f}s | "
                  f"{abs(float(np.mean(zn(x)*zn(y)))):13.3f} | {a:13.3f} {b:7.3f} {b-a:+7.3f}")
        print()
        fig=plt.figure(figsize=(17,13));gs=GridSpec(5,1,figure=fig,hspace=.32)
        for k in range(4):
            a=fig.add_subplot(gs[k]);x=F[k][off:off+SPAN];y=C[k][0][off:off+SPAN];f=C[k][1][off:off+SPAN]
            a.plot(t,zn(gw),color='k',lw=2.2,alpha=.55,label='BIOPAC GT')
            a.plot(t,zn(x),color='#95a5a6',lw=1.5,label='before')
            a.plot(t,zn(y),color=PCOL[k],lw=1.5,label='after (conditioned)')
            a.plot(t[f],zn(x)[f],'.',color='#c0392b',ms=3.5,label='flagged as transient')
            a.set_xlim(t[0],t[-1]);a.grid(alpha=.15);a.legend(fontsize=7,ncol=4,loc='upper right')
            a.set_ylabel(CAND[k],fontsize=9,color=PCOL[k],fontweight='bold')
            cb=abs(float(np.mean(zn(x)*zn(gw))));ca=abs(float(np.mean(zn(y)*zn(gw))))
            a.text(.005,.95,f'repaired {C[k][2]:.1%}   |corr| to GT  {cb:.3f} → {ca:.3f}',
                   transform=a.transAxes,ha='left',va='top',fontsize=8,color='#333',fontweight='bold')
            if k<3: a.set_xticklabels([])
        a=fig.add_subplot(gs[4])
        for LS,col,dy in [(4.0,'#16a085',0),(6.0,'#8e44ad',1)]:
            L=int(LS*FS);cen=[];val=[]
            for s in range(off,off+SPAN-L+1,L):
                cen.append((s-off+L/2)/FS+S0/FS)
                val.append(np.mean([C[k][1][s:s+L].mean() for k in range(4)]))
            a.bar(np.array(cen)+dy*0.6-0.3,val,width=LS*0.42,color=col,label=f'{LS:.0f} s chunks')
        a.set_xlim(t[0],t[-1]);a.grid(alpha=.15);a.legend(fontsize=8)
        a.set_ylabel('fraction repaired\n(mean over the 4)',fontsize=9,fontweight='bold')
        a.set_xlabel('time (s)')
        fig.suptitle(f'STEP 1 — chunk conditioner   {NM}  t={S0/FS:.0f}-{(S0+SPAN)/FS:.0f} s\n'
                     f'|robust z| > {THR} over {CTX/FS:.0f} s context, only runs shorter than '
                     f'{MAXDUR/FS:.1f} s (a breath is >= 1.7 s)   |   black = BIOPAC GT',
                     fontweight='bold',fontsize=13)
        pth=f'{MTR}/dit_step1_{NM}.png';fig.savefig(pth,dpi=115,bbox_inches='tight');plt.close(fig)
        print(f'  [saved] {pth}')
    d=np.array([x[3] for x in SUM]);fr=np.array([x[2] for x in SUM])
    print(f"\n=== over {len(SUM)} traces ({len(SUBS)} subjects x 4) ===")
    print(f"  repaired: mean {fr.mean():.1%}, median {np.median(fr):.1%}, max {fr.max():.1%}")
    print(f"  |corr| to GT change: mean {d.mean():+.4f}, median {np.median(d):+.4f}, "
          f"improved {np.mean(d>0):.0%} of traces")
    print("  (this is raw |corr|, not real-gain — conditioning is about smoothness for the next stage,")
    print("   not about beating the null. The number is here only to show it does not destroy the signal.)")
