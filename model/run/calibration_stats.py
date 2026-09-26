"""Calibration-statistics condition vector c_u = (amplitude-CV, interval-CV) for the DiT renderer (CALSTATS=1).

Both numbers are read DIRECTLY from a belt block (never from anything that passed through the model):
    interval-CV   sd/mean of the peak-to-peak intervals
    amplitude-CV  sd/mean of the peak-to-next-valley amplitudes on the band-passed, z-scored belt
Peaks/valleys come from the project-wide detector (metrics_timing.ev) with the local AGC fixed at
CU_AGC=9 s, independent of the process' AGC env - so the vector is the same number in training, in the bench and
in the scorer. Amplitudes are measured on the NON-AGC signal (the AGC would flatten exactly the variation we want).

Training side (train.py): every training row gets the vector of a RANDOM 84 s block of the same
subject that does not overlap the row's own 42 s window, re-drawn every epoch; z-scored with mean/sd over all such
blocks of the fold's TRAINING subjects only (stored as buffers cu_mu / cu_sd inside the weights).
Test side: the held-out subject's first 84 s of belt = the calibration block that
the rate anchor and the few-shot rows already use; eval windows start at >= 84 s.
"""
import _path  # noqa: F401  (see _path.py)
import os,numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed'
FS=17.0;RLO,RHI=0.1,0.6
CU_AGC=9.0            # seconds; the detector setting the count scorer uses (AGC=9)
CU_DIM=2;CU_NAMES=('ampCV','intCV')
CAL_S=84.0            # calibration block length (s), same as the bench's CALS default
MIN_PEAKS=4
def _zn(x): return (x-x.mean())/(x.std()+1e-9)
def _bpf(x):
    n=len(x);fr=np.fft.rfftfreq(n,1/FS);B=(fr>=RLO)&(fr<=RHI);x=x-x.mean()
    return np.fft.irfft(np.fft.rfft(x)*B,n=n)
def _agc(z,sec):
    r=np.sqrt(uniform_filter1d(z*z,max(3,int(sec*FS)),mode='nearest'));return z/np.maximum(r,0.25*np.sqrt(np.mean(z*z)+1e-12))
def _ev(z,sign):
    """same call as metrics_timing.ev with AGC=CU_AGC: prominence 0.4, min distance FS/RHI samples"""
    p,_=find_peaks(sign*_agc(z,CU_AGC),prominence=0.4,distance=int(FS/RHI));return p
def belt_stats(x):
    """(ampCV, intCV) of one belt block; NaN pair when the block has fewer than MIN_PEAKS peaks or amplitudes"""
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)<int(20*FS): return np.array([np.nan,np.nan])
    s=_zn(_bpf(x));pk=_ev(s,1);tr=_ev(s,-1)
    if len(pk)<MIN_PEAKS: return np.array([np.nan,np.nan])
    iv=np.diff(pk)/FS;intcv=iv.std()/iv.mean()
    amp=[]
    for p in pk:
        nx=tr[tr>p]
        if len(nx): amp.append(s[p]-s[nx[0]])
    amp=np.array(amp)
    ampcv=amp.std()/amp.mean() if len(amp)>=MIN_PEAKS-1 and amp.mean()>0 else np.nan
    return np.array([ampcv,intcv])
def belt_record(u):
    return np.load(f'{PRE}/sub{int(u):02d}/data_fc729.npz',allow_pickle=True)['gt_aligned'].astype(float)
def calib_stats(u,cal_s=CAL_S):
    """test-time vector: the subject's FIRST cal_s seconds of belt (the calibration block), nothing later"""
    return belt_stats(belt_record(u)[:int(cal_s*FS)])
def block_table(u,cal_s=CAL_S,hop_s=1.0):
    """(starts_s, stats (nb,2)) for every cal_s-long block of subject u's belt at a hop_s grid"""
    g=belt_record(u);L=int(cal_s*FS);n=len(g);starts=np.arange(0,n-L+1,int(hop_s*FS))
    st=np.stack([belt_stats(g[b:b+L]) for b in starts]) if len(starts) else np.zeros((0,2))
    return starts/FS,st
def valid_blocks(block_starts_s,row_start_s,row_len_s=42.0,cal_s=CAL_S):
    """blocks whose span [b, b+cal_s) does not intersect the row's [r, r+row_len_s)"""
    b=np.asarray(block_starts_s,float);return (b+cal_s<=row_start_s)|(b>=row_start_s+row_len_s)
