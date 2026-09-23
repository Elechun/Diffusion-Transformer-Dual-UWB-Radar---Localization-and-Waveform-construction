"""The main result tables: ACF, DTW, CSp, rate error and event-time errors, by timing-stability group.

Metric set fixed by the project's own choice rather than borrowed: ACF replaces the signed correlation as
the shape measure, and the reference papers' four - cosine, rate error, peak and valley time error - sit
beside it. F1 is kept as a secondary event measure. Two lengths: the 42 s route cycle, and the whole
held-out record.

Every metric carries a metronome row - a 20 bpm sine of random phase, which knows nothing about anybody.
That is the baseline the reference papers fill with a competing method; here the generator emits a
near-constant 19.95 bpm, so a constant-rate oscillator is the honest thing to beat. The metronome is
computed here for ALL metrics, not just the ones stored earlier, and it is computed on the same subjects
as the column it sits under, so a group mean is never compared against a cohort mean.

No top-N rows and no stranger-belt row. The two groups are fixed by alignment-lag scatter, a property of
the reference timing, and radar-side quality does not differ between them.

One caveat worth carrying: ACF compares two autocorrelation curves, which are blind to a sign flip and
largely blind to a shift. Since inversion and sub-second timing are this cohort's dominant failures, ACF
is systematically the most forgiving column here - it says whether the breathing rhythm was reproduced,
not whether it was placed correctly. The time-error columns are what answer placement.
"""
import os,glob,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu,ttest_rel
from metrics import zn,score
import diffusion_model as D
import metrics_windowed as T
import metrics_timing as PT
PRE='/home/user1/Desktop/UWB_BIOPAC/preprocessed';MTR='/home/user1/Desktop/UWB_BIOPAC/MTR'
OUT=os.environ.get('OUT') or f'{MTR}/main_tables'
TAG=os.environ.get('TAG','c40');IQR_TH=float(os.environ.get('IQR_TH','1.0'))
FS=D.FS;W=D.NC*D.CH
SCALES=[('42 s window',W),('whole subject',None)]
# (key, label, higher-is-better)
MET=[('ACF','ACF',True),('DTW','DTW',False),('CSp','CSp (cosine)',True),
     ('dRR','|d rate| (bpm)',False),('dtp','|dt| peak (s)',False),('dtv','|dt| valley (s)',False),
     ('mp','peaks matched %',True),('F1','breath F1',True)]

def allmetrics(a,b):
    """every metric this table reports, on one piece"""
    m=T.metrics(a,b);u=PT.unit(a,b)
    return dict(ACF=m['ACF'],DTW=m['DTW'],CSp=m['CSp'],F1=m['F1'],
                dRR=u['dRR'],dtp=u['dtp'],dtv=u['dtv'],mp=u['mp'])

def agg(rec,gt,L):
    n=1 if L is None else len(rec)//L
    acc={k:[] for k,_,_ in MET}
    for s in range(n):
        x,y=(rec,gt) if L is None else (rec[s*L:(s+1)*L],gt[s*L:(s+1)*L])
        if len(x)<int(8*FS) or np.std(y)<1e-9: continue
        v=allmetrics(x,y)
        for k in acc:
            if np.isfinite(v[k]): acc[k].append(v[k])
    return {k:(np.mean(v) if len(v) else np.nan) for k,v in acc.items()}

if __name__=='__main__':
    Z=[np.load(f,allow_pickle=True) for f in sorted(glob.glob(f'{PRE}/_br_arms_fc729_{TAG}_s*.npz'))]
    EV=Z[0]['eval_idx'];SE=D.S[EV];subs=np.array([int(u) for u in Z[0]['subs']])
    w=np.load(f'{PRE}/_worst_vs_best_{TAG}.npz',allow_pickle=True)
    assert [int(u) for u in w['subs']]==list(subs)
    g=w['lag_iqr']>IQR_TH
    print(f'  {TAG}, {len(Z)} seeds, {len(subs)} subjects, unstable n={int(g.sum())}',flush=True)
    mrs=np.random.RandomState(11)
    RES={}
    for sname,L in SCALES:
        own={k:[] for k,_,_ in MET};met={k:[] for k,_,_ in MET}
        for u in subs:
            kw=np.flatnonzero(SE==u);ao={k:[] for k in own};am={k:[] for k in met}
            for z in Z:
                if L is None:
                    rec=np.concatenate([zn(z['wv_0'][j]) for j in kw])
                    gt =np.concatenate([zn(D.G[EV[j]]) for j in kw])
                    tw=np.arange(len(kw)*W)/FS
                    mt=zn(np.sin(2*np.pi*(20.0/60)*tw+mrs.uniform(0,2*np.pi)))
                    for d,x in ((ao,rec),(am,mt)):
                        q=agg(x,gt,None)
                        for k in d: d[k].append(q[k])
                else:
                    vo={k:[] for k in own};vm={k:[] for k in met}
                    tt=np.arange(W)/FS
                    for j in kw:
                        gt=zn(D.G[EV[j]])
                        qo=agg(zn(z['wv_0'][j]),gt,L)
                        qm=agg(zn(np.sin(2*np.pi*(20.0/60)*tt+mrs.uniform(0,2*np.pi))),gt,L)
                        for k in vo: vo[k].append(qo[k]);vm[k].append(qm[k])
                    for k in ao: ao[k].append(np.nanmean(vo[k]));am[k].append(np.nanmean(vm[k]))
            for k in own: own[k].append(np.nanmean(ao[k]));met[k].append(np.nanmean(am[k]))
        RES[sname]=dict(own={k:np.array(v) for k,v in own.items()},
                        met={k:np.array(v) for k,v in met.items()})
        print(f'    {sname} done',flush=True)

    os.makedirs(OUT,exist_ok=True);plt.rcParams.update({'font.size':10.5})
    fig,axes=plt.subplots(len(SCALES),1,figsize=(13.2,5.0*len(SCALES)))
    for ai,(sname,_) in enumerate(SCALES):
        R=RES[sname];ax=axes[ai];ax.axis('off')
        cols=[f'stable\nn={int((~g).sum())}',f'unstable\nn={int(g.sum())}',
              f'all {len(subs)}','group p','metronome\n(all)','vs metronome\n(all)']
        cell=[];colr=[];rows=[]
        for k,lab,hi in MET:
            o=R['own'][k];m=R['met'][k]
            a=o[~g];b=o[g];ok=np.isfinite(a);ok2=np.isfinite(b)
            gp=mannwhitneyu(a[ok],b[ok2]).pvalue
            f=lambda x:f'{x:.2f}' if k in ('dRR','mp') else f'{x:.3f}'
            good=np.isfinite(o)&np.isfinite(m)
            tp=ttest_rel(o[good],m[good])[1]
            better=(np.nanmean(o)>np.nanmean(m)) if hi else (np.nanmean(o)<np.nanmean(m))
            rows.append(lab)
            cell.append([f(np.nanmean(a)),f(np.nanmean(b)),f(np.nanmean(o)),f'{gp:.4f}',
                         f(np.nanmean(m)),f'{"better" if better else "worse"}  p={tp:.4f}'])
            bs=(np.nanmean(a)>np.nanmean(b)) if hi else (np.nanmean(a)<np.nanmean(b))
            colr.append(['#d8efdf' if bs else '#f7dcd9','#f7dcd9' if bs else '#d8efdf',
                         '#ffffff','#ffffff','#f0f2f5',
                         '#d8efdf' if (better and tp<0.05) else ('#f7dcd9' if (not better and tp<0.05)
                                                                 else '#ffffff')])
        tb=ax.table(cellText=cell,rowLabels=rows,colLabels=cols,cellColours=colr,
                    loc='center',cellLoc='center')
        tb.auto_set_font_size(False);tb.set_fontsize(10.5);tb.scale(1,1.85)
        for (r,c),cl in tb.get_celld().items():
            if r==0 or c==-1: cl.set_text_props(weight='bold')
        ax.set_title(f'{sname}',fontweight='bold',fontsize=13,pad=14)
    fig.suptitle(f'{len(Z)} seeds, {len(subs)} subjects   -   groups fixed by alignment-lag scatter '
                 f'({IQR_TH:.1f} s), radar-side quality equal in both\n'
                 f'metronome = 20 bpm sine, random phase, computed on the same subjects',
                 fontweight='bold',fontsize=12.5,y=.995)
    fig.subplots_adjust(hspace=.30)
    p=f'{OUT}/main_tables.png';fig.savefig(p,dpi=122,bbox_inches='tight');plt.close(fig)
    for sname,_ in SCALES:
        R=RES[sname];print(f'\n  --- {sname} ---')
        print('  %-16s %9s %9s %9s %9s %9s'%('','stable','unstable','all','group p','metronome'))
        for k,lab,hi in MET:
            o=R['own'][k];m=R['met'][k];a=o[~g];b=o[g]
            gp=mannwhitneyu(a[np.isfinite(a)],b[np.isfinite(b)]).pvalue
            good=np.isfinite(o)&np.isfinite(m)
            print('  %-16s %9.3f %9.3f %9.3f %9.4f %9.3f   vs metro p=%.4f'%(
                lab,np.nanmean(a),np.nanmean(b),np.nanmean(o),gp,np.nanmean(m),
                ttest_rel(o[good],m[good])[1]))
    np.savez_compressed(f'{PRE}/_main_tables_{TAG}.npz',subs=subs,unstable=g,
                        **{f'{s}_{v}_{k}':RES[s][v][k] for s,_ in SCALES for v in RES[s]
                           for k in RES[s][v]})
    print(f'\n  {p}')
