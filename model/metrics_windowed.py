"""Every metric written under the waveform, piece by piece, so a good stretch and a bad one can be told apart.

One block per arm: the trace against the belt, and directly beneath it a grid with one column per piece and
one row per metric, coloured so the eye finds the bad pieces before reading any number.

    CS        cosine similarity, means left in - the raw overlap of the two traces
    Corr      Pearson, means removed - the same thing with any offset taken out
    ACF       correlation between the two autocorrelation curves. Blind to WHEN the breath started, so a
              stretch that is right but late still scores well here while CS and Corr punish it
    dBPM      breathing rate of the reconstruction minus the belt's, from the spectral peak in band
    F1        the breath-matching score, trough to trough

The piece is 14 s by default and that is deliberate. Over 3 s a stranger's belt already scores 0.553 on CS,
so a 3 s cell cannot tell a real match from a coincidence; and no peak counter can work inside a piece
shorter than one breath. SEG_S=3 or 6 or 21 or 42 changes it if the shorter view is wanted anyway.

The null for each metric - the same reconstruction against a different subject's window - is printed in the
row label, because none of these numbers means anything on its own.
"""
import os,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from metrics import zn,score
import diffusion_model as D
import signal_processing as RC
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed';MTR='/home/user1/Desktop/UWB_BIOPAC/MTR'
FS=D.FS;NC=D.NC;CH=D.CH;W=NC*CH
NPZ=os.environ.get('NPZ','_moe2_fc729_s0.npz')
SEG=int(float(os.environ.get('SEG_S','14'))*FS)
ARMS_SHOW=os.environ.get('ARMS','')                    # comma-separated indices, default all
TOPN=int(os.environ.get('TOPN','10'))
ROWS=['CSp','CS','DTW','ACF','dBPM','F1']
# SIGNED, not absolute. The absolute value scored a perfectly inverted reconstruction as a perfect one:
# a piece running exactly out of phase read 0.92 while the trace visibly rose where the belt fell. It also
# disagreed with training, which has always used a signed rho for exactly this reason - an absolute value
# hands the model a free sign flip per window. Across the cohort 31-37% of pieces are inverted, so this is
# not a corner case; taking the sign back roughly halves every CS and Corr ever reported here.
def cs(a,b):
    """Pearson, i.e. the cosine AFTER removing each mean - which is what the row is labelled.

    This used to be a bare cosine. On a whole 42 s window that made no difference, because the window had
    already been z-scored and its mean was exactly zero. On a 14 s slice OF that window it did: a slice of
    a zero-mean array is not zero-mean, so the DC term crept back in and the number stopped being a
    correlation. Measured across 720 slices the drift was small on average, +9.3e-05, but reached 1.8e-02
    on individual pieces. Removing the mean here makes the row mean the same thing at every length.

    csp below is a different matter and is deliberately left uncentred: that IS the papers' definition."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a-a.mean();b=b-b.mean()
    return float(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
def csp(a,b):
    """MoRe-Fi Eq 14 / BreathCatcher Eq 27, which are the same expression: no mean removed, on
    waveforms scaled to [0,1] as their figures' axes are. Two band-limited noises reach 0.874
    under it, so it is carried alongside the signed one rather than instead of it."""
    mm=lambda x:(np.asarray(x,float)-np.min(x))/(np.max(x)-np.min(x)+1e-12)
    a=mm(a);b=mm(b);return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
DTW_DEC=3
# The half-width of the Sakoe-Chiba band, in seconds. Default 0.5; set DTW_BAND_S to try another.
DTW_BAND_S=float(os.environ.get('DTW_BAND_S','0.5'))
DTW_BAND=max(1,int(round(DTW_BAND_S*FS/DTW_DEC)))   # +-0.5 s is a sixth of a breath
# The band was +-1.5 s and that was wrong: a breath lasts about 2.9 s, so a band over half a period could
# stretch an extra breath onto a missing one and call it a small shift. A piece running 6 bpm fast scored
# BETTER than one with the right rate. At +-0.5 s the ordering follows what the traces show.
def dtw(a,b,dec=DTW_DEC,band=DTW_BAND):
    """banded DTW cost per step between two z-scored pieces - lower is better.

    Correlation dies once the peaks are 0.6 s apart, which is 20% of a breath, but the standing diagnosis is
    that the breaths are right and late. DTW lets the two traces stretch against each other within a band and
    charges only for the shape that is left over. The band is what keeps it honest: without a limit anything
    can be warped onto anything.
    """
    x=a[:len(a)//dec*dec].reshape(-1,dec).mean(1);y=b[:len(b)//dec*dec].reshape(-1,dec).mean(1)
    x=(x-x.mean())/(x.std()+1e-9);y=(y-y.mean())/(y.std()+1e-9)
    n,m=len(x),len(y);INF=1e18
    prev=np.full(m+1,INF);prev[0]=0.
    for i in range(1,n+1):
        cur=np.full(m+1,INF)
        lo=max(1,i-band);hi=min(m,i+band)
        for j in range(lo,hi+1):
            c=(x[i-1]-y[j-1])**2
            cur[j]=c+min(prev[j],cur[j-1],prev[j-1])
        prev=cur
    return float(prev[m]/(n+m))
def pear(a,b): return float(np.corrcoef(a,b)[0,1])          # signed, see the note on cs above
def acf(x,mx):
    v=x-x.mean();m=1
    while m<2*len(x): m*=2
    f=np.fft.rfft(v,m);r=np.fft.irfft(f*np.conj(f),m)[:mx+1]
    return r/(r[0]+1e-12)
def bpm(x):
    """breathing rate from the spectral peak, interpolated between bins.

    Over 14 s one FFT bin is 4.29 breaths a minute, so the raw peak can only ever report a multiple of that
    - 0, +-4.3, +-8.6 - which is coarser than the differences being looked for. Fitting a parabola through
    the peak and its two neighbours in log power recovers a fraction of a bin.
    """
    v=(x-x.mean())*np.hanning(len(x));sp=np.abs(np.fft.rfft(v))**2
    fr=np.fft.rfftfreq(len(x),1/FS);m=np.flatnonzero((fr>=RC.RLO)&(fr<=RC.RHI))
    if not len(m): return np.nan
    k=m[np.argmax(sp[m])]
    d=0.0
    if 0<k<len(sp)-1:
        y0,y1,y2=np.log(sp[k-1]+1e-20),np.log(sp[k]+1e-20),np.log(sp[k+1]+1e-20)
        den=y0-2*y1+y2
        if den<-1e-12: d=float(np.clip(0.5*(y0-y2)/den,-0.5,0.5))
    return 60*float(fr[k]+d*(fr[1]-fr[0]))
def metrics(a,b):
    mx=min(int(8*FS),len(a)-1)
    r={'CSp':csp(a,b),'CS':cs(a,b),'DTW':dtw(a,b),'ACF':pear(acf(a,mx)[1:],acf(b,mx)[1:]),
       'dBPM':bpm(a)-bpm(b)}
    r['F1']=score(a,b,FS)['breath_f1'] if len(a)>=int(8*FS) else np.nan
    return r
COL={'CS':(0.10,0.45),'ACF':(0.30,0.85),'F1':(0.20,0.60)}
# CSp has no fixed floor: without mean removal its baseline depends on the LENGTH of the piece.
# Two band-limited noises reach 0.874 over 42 s but only ~0.81 over 14 s, so a fixed scale painted
# 0.85 red at 14 s when 0.85 was actually above that length's own floor. Filled from the measured
# stranger null, the same way DTW is.
CSPCOL=[0.81,0.95]
DTWCOL=[0.5,1.0]                                       # filled in from the null once it is measured
def shade(k,v):
    """green good, red bad. Kept in float throughout - an int index into a colormap means something else"""
    if not np.isfinite(v): return '#ffffff'
    if k=='DTW':
        lo,hi=DTWCOL;g=1.0-float(np.clip((v-lo)/(hi-lo),0.0,1.0))   # lower cost is better
    elif k=='CSp':
        lo,hi=CSPCOL;g=float(np.clip((v-lo)/(hi-lo),0.0,1.0))
    elif k=='dBPM':
        g=1.0-float(np.clip(abs(v)/6.0,0.0,1.0))       # 6 breaths a minute off is fully bad
    else:
        lo,hi=COL[k];g=float(np.clip((v-lo)/(hi-lo),0.0,1.0))
    return plt.cm.RdYlGn(g)
if __name__=='__main__':
    z=np.load(f'{PRE}/{NPZ}',allow_pickle=True)
    EV=z['eval_idx'];arms=[str(v) for v in z['arms']];na=len(arms);subs=z['subs'];SE=D.S[EV]
    CC=np.stack([z[f'cc_{i}'] if f'cc_{i}' in z else z[f'f1_{i}'] for i in range(na)])
    WV=[z[f'wv_{a}'] for a in range(na)];G=D.G
    idx=[int(v) for v in ARMS_SHOW.split(',')] if ARMS_SHOW else list(range(na))
    tag=NPZ.replace('.npz','').lstrip('_')
    rs=np.random.RandomState(4);n=len(EV)
    oth=np.array([rs.choice(np.flatnonzero(SE!=SE[j])) for j in range(n)])
    NUL={k:[] for k in ROWS}                            # what a stranger's window scores, same pieces
    for a in idx:
        for j in range(n):
            for s in range(W//SEG):
                sl=slice(s*SEG,(s+1)*SEG)
                m=metrics(zn(WV[a][j])[sl],zn(G[EV[oth[j]]])[sl])
                for k in ROWS: NUL[k].append(m[k])
    NULL={k:(np.nanmean(np.abs(NUL[k])) if k=='dBPM' else np.nanmean(NUL[k])) for k in ROWS}
    DTWCOL[0]=0.55*NULL['DTW'];DTWCOL[1]=1.05*NULL['DTW']   # the scale is set by what a stranger costs
    CSPCOL[0]=NULL['CSp'];CSPCOL[1]=max(0.95,NULL['CSp']+0.12)
    print('  null level of each metric (same output, a stranger\'s window):')
    print('   '+'   '.join(f'{k} {NULL[k]:.3f}' for k in ROWS)+'\n')
    sfx=('_arm'+ARMS_SHOW.replace(',','')) if ARMS_SHOW else ''
    od=os.environ.get('OUT') or f'{MTR}/{tag}_chunktable{sfx}'
    os.makedirs(od,exist_ok=True)
    order=np.argsort(-CC[0])[:TOPN]
    plt.rcParams.update({'font.size':9})
    for rank,si in enumerate(order,1):
        u=int(subs[si]);kw=np.flatnonzero(SE==u)
        if not len(kw): continue
        npc=len(kw)*(W//SEG)
        t=np.arange(len(kw)*W)/FS;gt=np.concatenate([zn(G[i]) for i in EV[kw]])
        fig=plt.figure(figsize=(max(19,npc*0.62),(2.05+1.35)*len(idx)+1.2))
        outer=GridSpec(len(idx),1,figure=fig,hspace=.30)      # room for each arm's title
        for r,a in enumerate(idx):
            gs=GridSpecFromSubplotSpec(2,1,subplot_spec=outer[r],hspace=.0,height_ratios=[2.05,1.35])
            rec=np.concatenate([zn(WV[a][j]) for j in kw])
            ax=fig.add_subplot(gs[0])
            ax.plot(t,gt,color='k',lw=1.3,alpha=.85,label='BIOPAC')
            ax.plot(t,rec,color='#c0392b',lw=1.05,label=arms[a])
            for q in range(1,len(kw)): ax.axvline(q*W/FS,color='#34495e',lw=1.1,ls='--',alpha=.6)
            for s in range(1,npc): ax.axvline(s*SEG/FS,color='#bdc3c7',lw=.6,alpha=.7)
            ax.set_xlim(0,len(kw)*W/FS);ax.set_ylim(-3.4,3.4);ax.grid(alpha=.12)
            ax.set_xticklabels([]);ax.legend(fontsize=8,loc='upper right',ncol=2)
            M=[metrics(rec[s*SEG:(s+1)*SEG],gt[s*SEG:(s+1)*SEG]) for s in range(npc)]
            good=np.mean([m['Corr'] for m in M])
            ax.set_title(f'{arms[a]}      window |corr| {CC[a,si]:.3f}      '
                         f'piece mean: CS {np.mean([m["CS"] for m in M]):.3f}  Corr {good:.3f}  '
                         f'DTW {np.mean([m["DTW"] for m in M]):.3f}  '
                         f'ACF {np.nanmean([m["ACF"] for m in M]):.3f}  '
                         f'|dBPM| {np.nanmean(np.abs([m["dBPM"] for m in M])):.2f}  '
                         f'F1 {np.nanmean([m["F1"] for m in M]):.3f}',fontweight='bold',fontsize=10)
            ax=fig.add_subplot(gs[1])
            for ri,k in enumerate(ROWS):
                for s in range(npc):
                    v=M[s][k]
                    ax.add_patch(plt.Rectangle((s*SEG/FS,len(ROWS)-1-ri),SEG/FS,1,
                                 facecolor=shade(k,v),edgecolor='white',lw=.6))
                    ax.text((s+.5)*SEG/FS,len(ROWS)-.5-ri,
                            ('--' if not np.isfinite(v) else (f'{v:+.1f}' if k=='dBPM' else f'{v:.2f}')),
                            ha='center',va='center',fontsize=7.5)
            ax.set_xlim(0,len(kw)*W/FS);ax.set_ylim(0,len(ROWS))
            ax.set_yticks(np.arange(len(ROWS))+.5)
            ax.set_yticklabels([f'{k}  (null {NULL[k]:.2f})' for k in ROWS[::-1]],fontsize=7.5)
            ax.set_xticks([]);ax.set_frame_on(False)
            if r==len(idx)-1:
                ax.set_xlabel(f'one cell = {SEG/FS:.0f} s   |   dashed line = 42 s window boundary   |   '
                              f'green good, red bad; dBPM red at 6 breaths a minute off',fontsize=8.5)
        fig.suptitle(f'Rank {rank}   sub{u:02d}   {tag}',fontweight='bold',fontsize=13,y=1.002)
        p=f'{od}/Rank{rank:02d}_sub{u:02d}_{tag}_table.png'
        fig.savefig(p,dpi=100,bbox_inches='tight');plt.close(fig)
        print(f'  {p}',flush=True)
    print(f'\n  -> {od}\n')
