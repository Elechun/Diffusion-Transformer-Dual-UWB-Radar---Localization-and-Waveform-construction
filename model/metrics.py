"""The scoring panel for waveform reconstruction, replacing correlation as the headline.

Correlation to the GT, even with its null subtracted, cannot be the main criterion for this task:
  - it is one scalar for a whole window, so "uniformly mediocre" and "half excellent, half broken" get the
    same number. Which half is broken is exactly what building the model needs to know.
  - at short windows it is the difference of two large numbers (the 3 s null is 0.92), so it is mostly noise.
  - a reconstruction with the right breath morphology but a slipped phase scores near zero, which is not
    what "failed to reconstruct" should mean.
But GT-free metrics alone cannot replace it either: their optimum is a pure sine wave - no spikes, no
spurious turns, maximal band concentration, and no relation to the subject's breathing. And the obvious
GT-referenced fix, counting how many GT turning points have a match nearby, was measured to rise when a
trace is merely smoothed - averaging a candidate with a phase-randomised partner raised it from 64.7% to
74.2%, beating the real partner.

So the panel below is built from GT-referenced measures that smoothing cannot inflate, each with a
phase-randomised surrogate control.

  breath F1        every GT breath (trough to trough) is matched ONE-TO-ONE with a reconstruction breath
                   within +-0.5 s. Smoothing removes reconstruction events, which costs RECALL; adding
                   noise creates them, which costs PRECISION. Reporting both closes the loophole that the
                   turning-point metric had.
  phase-locked     fraction of TIME the instantaneous phase (Hilbert, in-band) is within +-30 deg of the
                   GT's. Not an average - a fraction of samples, so a reconstruction that tracks for 20 s
                   and slips for 20 s reads 0.5 rather than a mid-range correlation.
                   The constant offset is removed only up to +-0.5 s, the sync tolerance. A first version
                   removed the MEDIAN phase difference, which is an unbounded lag: an unrelated sine at the
                   same rate then scored 0.98 and a phase-randomised surrogate 0.97, because what it
                   actually measured was agreement of RATE, not of phase.
  longest lock     the longest unbroken stretch, in seconds, where the phase stays inside that band. Two
                   reconstructions can have the same locked fraction with one tracking for 20 s straight
                   and the other flickering; only the first is a reconstruction.
                   (A per-breath local-null hit rate was tried and dropped: on a periodic signal every
                   breath resembles every other, so the local null saturates - a perfect reconstruction
                   scored 0.57 and a featureless single tone 0.71.)
  slip rate        cycles gained or lost per minute against the GT, from the unwrapped phase. A signed
                   number: positive means the reconstruction breathes faster than the subject.

Exports score(rec, gt, fs) -> dict, and surrogate_control(rec, gt, n) -> the same dict computed against
phase-randomised copies, so every number can be reported against what shape alone would give.
"""
import numpy as np
from scipy.signal import find_peaks, hilbert, butter, filtfilt
FS_DEF=17.0;RLO,RHI=0.1,0.6
zn=lambda x:(x-np.mean(x))/(np.std(x)+1e-12)
def _band(x,fs):
    b,a=butter(3,[RLO/(fs/2),RHI/(fs/2)],btype='band');return filtfilt(b,a,x)
def breaths(x,fs):
    """trough-to-trough breath boundaries; prominence tied to the trace's own scale so it is scale free.
    The minimum interval is 1/RHI = 1.67 s, not a tuned number: extract_hz ends in a 0.1-0.6 Hz bandpass, so
    a breath shorter than 1/0.6 s cannot be present and any detection at that spacing is noise. The earlier
    1.2 s setting let the detector find events its own passband forbids, which inflated both the real and
    the surrogate arm and exaggerated the gap between them - on the raw candidates it moved COM-LOS from
    -0.0051 (p=0.44) to -0.0127 (p=0.046), i.e. it manufactured a significant negative."""
    z=zn(x);pk,_=find_peaks(-z,prominence=0.4,distance=int(fs/RHI))
    return pk
def _match(a,b,tol):
    """greedy one-to-one nearest matching; returns matched pairs"""
    if not len(a) or not len(b): return []
    D=np.abs(a[:,None]-b[None,:]);pairs=[]
    used_a=set();used_b=set()
    order=np.dstack(np.unravel_index(np.argsort(D,axis=None),D.shape))[0]
    for i,j in order:
        if D[i,j]>tol: break
        if i in used_a or j in used_b: continue
        used_a.add(i);used_b.add(j);pairs.append((i,j))
    return pairs
def score(rec,gt,fs=FS_DEF,tol_s=0.5,phase_tol_deg=30.0):
    rec=np.asarray(rec,float);gt=np.asarray(gt,float)
    n=min(len(rec),len(gt));rec,gt=rec[:n],gt[:n]
    out={}
    tol=tol_s*fs
    bg=breaths(gt,fs);br=breaths(rec,fs)
    pairs=_match(bg.astype(float),br.astype(float),tol)
    rec_n=max(len(br),1);gt_n=max(len(bg),1)
    prec=len(pairs)/rec_n;recl=len(pairs)/gt_n
    out['breath_precision']=prec;out['breath_recall']=recl
    out['breath_f1']=0.0 if prec+recl==0 else 2*prec*recl/(prec+recl)
    out['breath_dt']=float(np.median([abs(bg[i]-br[j])/fs for i,j in pairs])) if pairs else np.nan
    pr=np.unwrap(np.angle(hilbert(_band(zn(rec),fs))))
    pg=np.unwrap(np.angle(hilbert(_band(zn(gt),fs))))
    d0=pr-pg
    # remove a constant offset only up to the sync tolerance, not an unbounded one
    f0=np.median(np.abs(np.diff(pg)))*fs/(2*np.pi)+1e-9        # GT rate in Hz
    cap=2*np.pi*f0*0.5                                          # phase equivalent of 0.5 s
    off=float(np.clip(np.median(d0),-cap,cap))
    d=d0-off
    err=np.abs(np.angle(np.exp(1j*d)))
    ok=err<np.deg2rad(phase_tol_deg)
    out['phase_locked']=float(np.mean(ok))
    if ok.any():
        e=np.diff(np.concatenate([[0],ok.astype(int),[0]]))
        runs=(np.flatnonzero(e<0)-np.flatnonzero(e>0))
        out['longest_lock_s']=float(runs.max()/fs)
    else:
        out['longest_lock_s']=0.0
    out['slip_per_min']=float((d[-1]-d[0])/(2*np.pi)/(n/fs)*60.0)
    out['n_breaths_gt']=len(bg);out['n_breaths_rec']=len(br)
    return out
def surrogate_control(rec,gt,fs=FS_DEF,n=8,seed=0):
    """the same panel, with the reconstruction replaced by a phase-randomised copy of itself.
    Same spectrum, same smoothness, no timing - whatever a metric gives here is what shape alone buys."""
    rs=np.random.RandomState(seed);acc=[]
    for _ in range(n):
        F=np.fft.rfft(rec);ph=rs.uniform(0,2*np.pi,len(F));ph[0]=0
        s=np.fft.irfft(np.abs(F)*np.exp(1j*ph),len(rec))
        acc.append(score(s,gt,fs))
    return {k:float(np.nanmean([a[k] for a in acc])) for k in acc[0]}
if __name__=='__main__':
    # sanity: the panel must punish both smoothing and noise, which is the property the old metric lacked
    rs=np.random.RandomState(0);fs=FS_DEF;t=np.arange(int(60*fs))/fs
    gt=np.sin(2*np.pi*0.25*t)+0.15*np.sin(2*np.pi*0.5*t)
    cases={'perfect':gt.copy(),
           'phase slipped 0.3 Hz':np.sin(2*np.pi*0.30*t),
           'smoothed to one tone':np.sin(2*np.pi*0.25*t),
           'noisy':zn(gt)+1.2*_band(rs.randn(len(t)),fs)*3,
           'unrelated':np.sin(2*np.pi*0.25*t+2.1)}
    print(f'{"case":22s} {"F1":>6s} {"prec":>6s} {"recall":>7s} {"phase-lock":>11s} {"longest(s)":>11s} {"slip/min":>9s}')
    for k,v in cases.items():
        s=score(v,gt,fs)
        print(f'{k:22s} {s["breath_f1"]:6.2f} {s["breath_precision"]:6.2f} {s["breath_recall"]:7.2f} '
              f'{s["phase_locked"]:11.2f} {s["longest_lock_s"]:11.1f} {s["slip_per_min"]:+9.2f}')
    sc=surrogate_control(zn(gt)+0.5*_band(rs.randn(len(t)),fs)*3,gt,fs)
    print(f'\nsurrogate of a noisy copy (same spectrum, no timing): F1 {sc["breath_f1"]:.2f}  '
          f'phase-lock {sc["phase_locked"]:.2f}  longest {sc["longest_lock_s"]:.1f} s')
    print('a metric is usable only if the real cases clear its own surrogate.')
