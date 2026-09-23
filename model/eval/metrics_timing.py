import _path  # noqa: F401  (see _path.py)
import os
"""The two BreathCatcher / MoRe-Fi error metrics, at three lengths and four subject subsets.

    Time estimation error       |t_p^e - t_p^a| and |t_v^e - t_v^a|, the absolute difference between the
                                estimated and the actual peak time, and the same for valleys
    Rate estimation error       |rho^e - rho^a|, from the strongest component of the frequency spectrum

Two things the papers do not state, so they are decided here and printed on the figure:

    matching    the papers write "the" peak and "the" valley, but a 42 s trace has many of each. Each
                actual event is paired with the nearest estimated event of the same kind, one to one,
                greedily. The pairing is refused beyond half the actual breath period, because past that
                point the nearest estimate belongs to a different breath and the number stops meaning
                "how late was this breath" and starts meaning "which breath did I land on"

    coverage    an error can only be reported for events that were paired at all. A trace that produces
                three peaks where the belt has fifteen will show a small error on those three. So the
                matched fraction is carried next to every error, and no error should be read without it

Peaks and valleys use the same detector as everything else in this project: prominence 0.4 on the
z-scored trace, minimum spacing 1/0.6 s, which is the shortest breath the 0.1-0.6 Hz passband allows.

Polarity is not corrected. If a reconstruction is inverted its peaks land on the belt's valleys, and the
metric is supposed to say so. The size of that effect is printed separately.

Two floors are carried, and the second one is the one that matters here:

    stranger belt   this reconstruction against a different subject's belt
    metronome       a plain 20 bpm sine with a random phase, which knows nothing about anybody. The
                    generator was measured to emit a near-constant 19.95 bpm (sd 0.34 against the belt's
                    3.71), so a constant-rate oscillator is what it has to beat before a time or rate
                    number can be attributed to the radar at all
"""
import os,numpy as np
from scipy import stats
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from metrics import zn,_match
import diffusion_model as D
import gt_audit as GU
import metrics_windowed as T
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed';MTR='/home/user1/Desktop/UWB_BIOPAC/MTR'
FS=D.FS;W=D.NC*D.CH;RHI=0.6
NPZ=os.environ.get('NPZ','_br_arms_fc729_s%d.npz')
SEEDS=[int(s) for s in os.environ.get('SEEDS','0,1,2').split(',')]
ARM=int(os.environ.get('ARM','3'))
OUT=os.environ.get('OUT') or f'{MTR}/paper_metrics'
SCALES=[('14 s piece',238),('42 s window',W),('whole subject',None)]
SUBSETS=[15,20,30,None]
KEYS=['dtp','dtv','dRR','mp','mv']
LAB={'dtp':'|dt| peak  (s)','dtv':'|dt| valley  (s)','dRR':'|d rate|  (bpm)',
     'mp':'peaks matched %','mv':'valleys matched %'}
LOWER=('dtp','dtv','dRR')                      # for these, smaller is better

AGC=float(os.environ.get('AGC','0'))    # >0 : divide by the running RMS over AGC seconds before picking peaks, so a
                                        # quiet stretch is not silently dropped by the global-sd prominence threshold.
                                        # Applied identically to belt and model, so comparisons stay fair. 0 = old detector.
def _agc(x,sec):
    from scipy.ndimage import uniform_filter1d
    z=zn(x);r=np.sqrt(uniform_filter1d(z*z,max(3,int(sec*FS)),mode='nearest'));return z/np.maximum(r,0.25*np.sqrt(np.mean(z*z)+1e-12))
def ev(x,sign):
    """peak (sign=+1) or valley (sign=-1) sample indices, same detector used project-wide"""
    z=_agc(x,AGC) if AGC>0 else zn(x);p,_=find_peaks(sign*z,prominence=0.4,distance=int(FS/RHI));return p.astype(float)

def unit(rec,gt):
    """the two paper errors on one piece; returns nan where the belt gives nothing to compare against"""
    o={k:np.nan for k in KEYS}
    for tag,sign in (('p',1),('v',-1)):
        a=ev(gt,sign);b=ev(rec,sign)
        if len(a)<2: continue
        per=np.median(np.diff(a))                       # actual breath period, in samples
        pr=_match(a,b,0.5*per)
        o['m'+tag]=100.0*len(pr)/len(a)
        if pr: o['dt'+tag]=float(np.mean([abs(a[i]-b[j]) for i,j in pr])/FS)
    o['dRR']=abs(T.bpm(rec)-T.bpm(gt))
    return o

def agg(rec,gt,L):
    n=1 if L is None else len(rec)//L
    acc={k:[] for k in KEYS}
    for s in range(n):
        a,b=(rec,gt) if L is None else (rec[s*L:(s+1)*L],gt[s*L:(s+1)*L])
        if len(a)<int(8*FS) or np.std(b)<1e-9: continue
        u=unit(a,b)
        for k in acc:
            if np.isfinite(u[k]): acc[k].append(u[k])
    return {k:(np.mean(v) if len(v) else np.nan) for k,v in acc.items()}

if __name__=='__main__':
    Z=[np.load(f'{PRE}/{NPZ%s}',allow_pickle=True) for s in SEEDS]
    arms=[str(a) for a in Z[0]['arms']];name=arms[ARM]
    GT,did=GU.for_run(Z[0])
    if did: print(f'  scored against {len(did)} hand-aligned belts',flush=True)
    EV=Z[0]['eval_idx'];SE=D.S[EV];subs=Z[0]['subs'];n=len(EV)
    rs=np.random.RandomState(23);oth=np.array([rs.choice(np.flatnonzero(SE!=SE[j])) for j in range(n)])
    mrs=np.random.RandomState(11);tt=np.arange(W)/FS
    metro=lambda: zn(np.sin(2*np.pi*(20.0/60)*tt+mrs.uniform(0,2*np.pi)))
    print(f'  arm "{name}"   {len(SEEDS)} seeds   {n} windows   {len(subs)} subjects\n',flush=True)
    RES={}
    for sname,L in SCALES:
        per={v:{k:[] for k in KEYS} for v in ('own','stranger','metro')}
        for u in subs:
            kw=np.flatnonzero(SE==u);acc={v:{k:[] for k in KEYS} for v in per}
            for z in Z:
                if L is None:
                    rec=np.concatenate([zn(z[f'wv_{ARM}'][j]) for j in kw])
                    gt =np.concatenate([zn(GT[EV[j]])       for j in kw])
                    st =np.concatenate([zn(GT[EV[oth[j]]])  for j in kw])
                    # one continuous sine across the whole record - concatenating independent 42 s
                    # sines would put a phase jump at every seam and smear the baseline's own spectrum
                    tw=np.arange(len(kw)*W)/FS
                    mt =zn(np.sin(2*np.pi*(20.0/60)*tw+mrs.uniform(0,2*np.pi)))
                    for v,a,b in (('own',rec,gt),('stranger',rec,st),('metro',mt,gt)):
                        m=agg(a,b,None)
                        for k in acc[v]: acc[v][k].append(m[k])
                else:
                    for j in kw:
                        for v,a,b in (('own',zn(z[f'wv_{ARM}'][j]),zn(GT[EV[j]])),
                                      ('stranger',zn(z[f'wv_{ARM}'][j]),zn(GT[EV[oth[j]]])),
                                      ('metro',metro(),zn(GT[EV[j]]))):
                            m=agg(a,b,L)
                            for k in acc[v]: acc[v][k].append(m[k])
            for v in per:
                for k in per[v]: per[v][k].append(np.nanmean(acc[v][k]))
        RES[sname]={v:{k:np.array(per[v][k]) for k in per[v]} for v in per}
        print(f'    {sname} done',flush=True)

    os.makedirs(OUT,exist_ok=True);plt.rcParams.update({'font.size':9.5})
    fig,axes=plt.subplots(len(SCALES),1,figsize=(13.0,4.2*len(SCALES)))
    for ai,(sname,L) in enumerate(SCALES):
        R=RES[sname];ax=axes[ai];ax.axis('off')
        cell=[];colr=[];lab=[]
        for N in SUBSETS:
            lab.append(f'best {N} subjects' if N else f'all {len(subs)}')
            cr=[];cc=[]
            for k in KEYS:
                v=R['own'][k]
                o=np.argsort(v) if k in LOWER else np.argsort(-v)
                idx=o[:N] if N else o
                m=np.nanmean(v[idx])
                cr.append(f'{m:.3f}' if k in ('dtp','dtv') else f'{m:.2f}')
                f=np.nanmean(R['metro'][k])                      # shaded against the metronome
                better=(m<f) if k in LOWER else (m>f)
                d=abs(m-f)/(abs(f)+1e-9)
                cc.append(plt.cm.RdYlGn(0.5+(0.45 if better else -0.45)*min(1.0,d/0.25)))
            cell.append(cr);colr.append(cc)
        for tag,v in (('metronome 20 bpm (floor)','metro'),('stranger belt','stranger')):
            lab.append(tag);cr=[];cc=[]
            for k in KEYS:
                m=np.nanmean(R[v][k])
                cr.append(f'{m:.3f}' if k in ('dtp','dtv') else f'{m:.2f}')
                cc.append('#e3e9ef' if v=='metro' else '#f4f5f6')
            cell.append(cr);colr.append(cc)
        # paired subject-level test of all 41 against the metronome
        sig=[]
        for k in KEYS:
            a=R['own'][k];b=R['metro'][k];g=np.isfinite(a)&np.isfinite(b)
            p=stats.ttest_rel(a[g],b[g])[1]
            better=(a[g].mean()<b[g].mean()) if k in LOWER else (a[g].mean()>b[g].mean())
            sig.append(f'{"better" if better else "worse"}  p={p:.4f}')
        lab.append('all 41 vs metronome');cell.append(sig);colr.append(['#ffffff']*len(KEYS))
        tb=ax.table(cellText=cell,rowLabels=lab,colLabels=[LAB[k] for k in KEYS],
                    cellColours=colr,loc='center',cellLoc='center')
        tb.auto_set_font_size(False);tb.set_fontsize(9.5);tb.scale(1,1.6)
        for (r,c),cl in tb.get_celld().items():
            if r==0 or c==-1: cl.set_text_props(weight='bold')
        ax.set_title(f'{sname}   -   each column ranked by itself, so "best 15" is a different 15 people '
                     f'in each column;  cells shaded against the metronome row',
                     fontweight='bold',fontsize=11,pad=14)
    fig.suptitle(f'{name}   -   {len(SEEDS)} seeds, {len(subs)} subjects\n'
                 f'time and rate estimation error as defined in BreathCatcher / MoRe-Fi.  Events paired '
                 f'nearest-first, one to one, refused beyond half a breath period;\n'
                 f'the matched percentage is what the error was averaged over - read the two together.  '
                 f'Polarity is not corrected.',fontweight='bold',fontsize=12,y=.998)
    fig.subplots_adjust(hspace=.40)
    p=f'{OUT}/paper_time_rate.png';fig.savefig(p,dpi=115,bbox_inches='tight');plt.close(fig)

    for sname,_ in SCALES:
        R=RES[sname];print(f'\n  --- {sname} ---')
        print('    '+'subset'.ljust(24)+''.join(f'{LAB[k]:>20s}' for k in KEYS))
        for N in SUBSETS:
            row=[]
            for k in KEYS:
                v=R['own'][k];o=np.argsort(v) if k in LOWER else np.argsort(-v)
                row.append(np.nanmean(v[o[:N] if N else o]))
            print('    '+(f'best {N}' if N else f'all {len(subs)}').ljust(24)+''.join(f'{x:20.3f}' for x in row))
        for v in ('metro','stranger'):
            print('    '+v.ljust(24)+''.join(f'{np.nanmean(R[v][k]):20.3f}' for k in KEYS))
        print('    '+'p (all 41 vs metro)'.ljust(24)+''.join(
            f'{stats.ttest_rel(R["own"][k][np.isfinite(R["own"][k])&np.isfinite(R["metro"][k])],R["metro"][k][np.isfinite(R["own"][k])&np.isfinite(R["metro"][k])])[1]:20.4f}' for k in KEYS))
    print(f'\n  {p}')
    np.savez_compressed(f'{PRE}/_paper_time_rate.npz',subs=subs,
                        **{f'{s}_{v}_{k}':RES[s][v][k] for s,_ in SCALES for v in RES[s] for k in KEYS})
