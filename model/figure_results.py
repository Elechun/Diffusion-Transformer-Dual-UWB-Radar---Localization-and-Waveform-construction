"""Final results package from the rev-belt csvs (no regeneration, no GPU).
  A  table_per_subject_rev.png      per-subject table, deployed system, reviewed belt
  B  bars_arms_rev.png              arm x metric bars, mean +- sd over 40 subjects, own vs radar-permuted
  D  dumbbell_own_vs_perm_rev.png   per-subject own vs perm, CS and timing (the radar contribution, subject by subject)
  E  coverage_scatter_rev.png       rate error vs CS with the coverage box
  F  bars_per_radar_perm_rev.png    deployed own vs both / COM-only / TV-only permutation
Run: python3 figure_results.py"""
import os,csv,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
M='/home/user1/Desktop/UWB_BIOPAC/MTR/rate_recovered'
W=[r for r in csv.DictReader(open(f'{M}/whole_main_table_reviewed_belt.csv')) if r['subject']!='mean']
P=list(csv.DictReader(open(f'{M}/rescore_rev_all_arms_per_subject.csv')))
f=lambda rows,k:np.array([float(r[k]) for r in rows])
def arm(a,which):
    d={r['subject']:r for r in P if r['arm']==a and r['which']==which};return d
# subject-wise 5-fold membership, reproduced from the training scripts (RandomState(0).permutation)
_subs=np.array(sorted({int(r['subject'][3:]) for r in P if r['which']=='own'}))
_pmx=np.random.RandomState(0).permutation(len(_subs))
FOLDS=[set(_subs[_pmx[i::5]].tolist()) for i in range(5)]
def foldstat(d,k):
    """mean over the 5 held-out folds and the sd BETWEEN folds: how much the estimate moves with which
    8 subjects were held out. The per-subject sd is a property of the cohort, not of the model, and is
    reported in the per-subject table instead."""
    v={int(u[3:]):float(d[u][k]) for u in d if np.isfinite(float(d[u][k]))}
    fm=np.array([np.mean([v[u] for u in f if u in v]) for f in FOLDS if any(u in v for u in f)])
    return fm.mean(),fm.std(ddof=1)
# ---------------- A. per-subject table ----------------
W.sort(key=lambda r:r['subject'])
Pd={r['subject']:r for r in P if r['arm']=='DiT deployed' and r['which']=='own'}
cols=['subject','belt bpm','model bpm','|Δbpm|','CS','ACF','peak %','valley %','time err (s)','pass']
rows=[];Q=[]
for r in W:
    q=Pd[r['subject']];Q.append(q);ok=float(q['RRAE_rev'])<=1.5 and float(q['CSp'])>=0.85
    rows.append([r['subject'][3:],f"{float(r['belt_bpm_rev']):.1f}",f"{float(r['model_bpm']):.1f}",f"{float(q['RRAE_rev']):.2f}",
                 f"{float(q['CSp']):.3f}",f"{float(q['ACF']):.3f}",f"{float(q['match_rev']):.1f}",f"{float(q['match_v']):.1f}",f"{float(q['te_rev']):.2f}",'✓' if ok else '—'])
npass=sum(r[-1]=='✓' for r in rows)
mean=['mean','','',f"{f(Q,'RRAE_rev').mean():.2f}",f"{f(Q,'CSp').mean():.3f}",f"{f(Q,'ACF').mean():.3f}",f"{f(Q,'match_rev').mean():.1f}",f"{f(Q,'match_v').mean():.1f}",f"{f(Q,'te_rev').mean():.2f}",f'{npass}/40']
fig,ax=plt.subplots(figsize=(10.5,12.6));ax.axis('off')
tb=ax.table(cellText=rows+[mean],colLabels=cols,loc='center',cellLoc='center');tb.auto_set_font_size(False);tb.set_fontsize(8.2);tb.scale(1,1.12)
for (i,j),c in tb.get_celld().items():
    if i==0: c.set_facecolor('#dfe7f1');c.set_text_props(weight='bold')
    elif i==len(rows)+1: c.set_facecolor('#f0f0f0');c.set_text_props(weight='bold')
    elif rows[i-1][-1]!='✓': c.set_facecolor('#fbeaea')
    if i>0 and i<=len(rows):
        if j==3 and float(rows[i-1][3])>1.5: c.set_text_props(color='#b3271e',weight='bold')
        if j==4 and float(rows[i-1][4])<0.85: c.set_text_props(color='#b3271e',weight='bold')
ax.set_title('Deployed system, whole walking course after the 84 s calibration, 3-seed mean per subject\n'
             'reference = human-reviewed belt peaks (rev); pass = rate error ≤ 1.5 bpm and CS ≥ 0.85; red rows fail',fontsize=10)
plt.savefig(f'{M}/table_per_subject_rev.png',dpi=140,bbox_inches='tight');plt.close()
# ---------------- B. bars with sd, own vs perm ----------------
ARMS=[('DiT deployed','DiT (ours)'),('TCN+adapt+anchor','TCN'),('U-Net+adapt+anchor','U-Net'),('Transformer+adapt+anchor','Transformer'),('IQ-VED+adapt+anchor','IQ-VED'),('RF-Carer(their SP+net)','RF-Carer\n(their SP+net)')]
MET=[('CSp','CS',(0.80,0.96)),('ACF','ACF',(0.70,0.96)),('RRAE_rev','|Δbpm| (bpm)',(0,3.0)),
     ('match_rev','peak match (%)',(80,96)),('match_v','valley match (%)',(80,96)),('te_rev','time estimation error (s)',(0.40,0.75))]
PERMBARS=os.environ.get('PERMBARS','0')=='1'   # 1: also draw the radar-permuted control beside each arm
fig,axs=plt.subplots(2,3,figsize=(17,9));axs=axs.ravel()
for ax,(k,lab,yl) in zip(axs,MET):
    x=np.arange(len(ARMS));wd=0.38 if PERMBARS else 0.62
    series=[(-wd/2,'own'),(wd/2,'perm')] if PERMBARS else [(0.0,'own')]
    for off,which in series:
        mu=[];sd=[]
        for a,_ in ARMS:
            m_,s_=foldstat(arm(a,which),k);mu.append(m_);sd.append(s_)
        if which=='own':
            cols=['#b3271e']+(['#9aa7b4']*(len(ARMS)-1) if not PERMBARS else ['#b3271e']*(len(ARMS)-1))
        else:
            cols=['#e8a9a3' if a=='DiT deployed' else '0.75' for a,_ in ARMS]
        ax.bar(x+off,mu,wd,yerr=sd,capsize=3,color=cols,edgecolor='k',lw=0.5,
               label=('own radar' if which=='own' else 'radar permuted') if (PERMBARS and k=='CSp') else None,
               error_kw=dict(lw=0.8))
        if which=='own':
            for xi,m in zip(x+off,mu):
                ax.text(xi,m+(0.004 if yl[1]<=1 else 0.4 if yl[1]>10 else 0.05),
                        f'{m:.3f}' if yl[1]<=1 else f'{m:.1f}' if yl[1]>10 else f'{m:.2f}',ha='center',fontsize=7.5)
    ax.set_xticks(x);ax.set_xticklabels([l for _,l in ARMS],fontsize=7.5,rotation=18)
    ax.set_ylim(*yl);ax.set_ylabel(lab);ax.grid(axis='y',alpha=.25)
    if PERMBARS and k=='CSp': ax.legend(fontsize=8,loc='upper right')
cap=('Same data, same folds, same calibration adaptation for every arm (regressors and RF-Carer: few-shot on the calibration block + one anchor projection); whole walking course, reviewed belt.\n'
     'Bars: mean over the 5 held-out folds of the subject-wise cross-validation. Error bars: sd BETWEEN folds, i.e. how much the estimate moves with which 8 subjects were held out.\n'
     'The much larger spread BETWEEN subjects is a property of the cohort rather than of the model and is given per subject in Table 1.\n'
     "RF-Carer uses its published signal-process layer and network on our recordings, so these are not that paper's reported numbers.")
if PERMBARS: cap+='\nPale/grey = the same model re-run with the radar conditioning taken from a window >=72 s away.'
fig.suptitle(cap,fontsize=9)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(f"{M}/bars_arms_rev{'_perm' if PERMBARS else ''}.png",dpi=140);plt.close()
# ---------------- D. dumbbell own vs perm ----------------
o=arm('DiT deployed','own');p=arm('DiT deployed','perm');us=sorted(set(o)&set(p))
fig,axs=plt.subplots(1,2,figsize=(13,8.5))
for ax,(k,lab,better) in zip(axs,(('CSp','CS','higher'),('deg_rev','peak timing error (deg)','lower'))):
    a=np.array([float(o[u][k]) for u in us]);b=np.array([float(p[u][k]) for u in us]);d=a-b;idx=np.argsort(d if better=='higher' else -d)
    y=np.arange(len(us))
    ax.hlines(y,b[idx],a[idx],color='0.75',lw=2);ax.scatter(b[idx],y,s=28,color='0.45',label='radar permuted',zorder=3);ax.scatter(a[idx],y,s=28,color='#b3271e',label='own radar',zorder=4)
    ax.set_yticks(y);ax.set_yticklabels([us[i][3:] for i in idx],fontsize=7.5);ax.invert_yaxis();ax.grid(axis='x',alpha=.25);ax.set_xlabel(lab)
    pv=wilcoxon(a,b).pvalue;n_b=int(((d>0) if better=='higher' else (d<0)).sum())
    ax.set_title(f'{lab}: own − perm = {d.mean():+.3f}, p = {pv:.3f} (Wilcoxon, n={len(us)}); own better in {n_b}/{len(us)}',fontsize=9.5);ax.legend(fontsize=8,loc='lower right')
fig.suptitle('What the radar contributes, subject by subject: same seed, same start noise, same anchor and few-shot; only the radar conditioning is taken from a window ≥72 s away',fontsize=10)
plt.tight_layout(rect=[0,0,1,0.95]);plt.savefig(f'{M}/dumbbell_own_vs_perm_rev.png',dpi=140);plt.close()
# ---------------- E. coverage scatter ----------------
rr=f(W,'RRAE_rev');cs=f(W,'CSp');ok=(rr<=1.5)&(cs>=0.85)
fig,ax=plt.subplots(figsize=(7.2,5.6))
ax.axvspan(0,1.5,ymin=0,ymax=1,color='#eaf3ea',zorder=0);ax.axhline(0.85,color='0.5',lw=0.8,ls='--');ax.axvline(1.5,color='0.5',lw=0.8,ls='--')
ax.scatter(rr[ok],cs[ok],s=42,color='#2e7d32',label=f'pass ({ok.sum()})',zorder=3);ax.scatter(rr[~ok],cs[~ok],s=42,color='#b3271e',label=f'fail ({(~ok).sum()})',zorder=3)
for r,c,s_ in zip(rr,cs,W):
    if not ((r<=1.5)&(c>=0.85)): ax.annotate(s_['subject'][3:],(r,c),fontsize=7.5,xytext=(4,3),textcoords='offset points')
ax.set_xlabel('rate error (bpm), reviewed belt');ax.set_ylabel('CS');ax.set_xlim(0,max(3.2,rr.max()+0.2));ax.set_ylim(0.78,0.98);ax.grid(alpha=.25);ax.legend(fontsize=9)
ax.set_title(f'Coverage: {ok.sum()}/40 subjects meet rate error ≤ 1.5 bpm and CS ≥ 0.85\nfailures are labelled; most lie right of the rate line (anchor rate drift after calibration)',fontsize=9.5)
plt.tight_layout();plt.savefig(f'{M}/coverage_scatter_rev.png',dpi=140);plt.close()
# ---------------- F. dual vs single radar ----------------
ARMS_R=[('DiT deployed','both radars\n(deployed)','#b3271e'),
        ('COM radar only','COM only','#5c6670'),
        ('TV radar only','TV only','#8a939c')]
MET_R=[('deg_rev','peak timing error (deg)',(70,84)),('te_rev','time estimation error (s)',(0.45,0.70)),
       ('match05','breaths matched within \u00b10.5 s (%)',(34,48))]
fig,axs=plt.subplots(1,3,figsize=(13.0,4.4))
for ax,(k,lab,yl) in zip(axs,MET_R):
    mu=[];sd=[];cols=[]
    for a,_,c in ARMS_R:
        m_,s_=foldstat(arm(a,'own'),k);mu.append(m_);sd.append(s_);cols.append(c)
    x=np.arange(len(ARMS_R))
    ax.bar(x,mu,0.60,yerr=sd,capsize=3.5,color=cols,edgecolor='k',lw=0.6,error_kw=dict(lw=0.9))
    for xi,m,s_ in zip(x,mu,sd):
        ax.text(xi,m+s_+(yl[1]-yl[0])*0.040,f'{m:.3f}' if yl[1]<=1 else f'{m:.1f}',
                ha='center',fontsize=10.5,weight='bold')
    ax.set_xticks(x);ax.set_xticklabels([l for _,l,_ in ARMS_R],fontsize=8.5)
    ax.set_ylim(*yl);ax.set_ylabel(lab,fontsize=10);ax.grid(axis='y',alpha=.25)
fig.suptitle('Breath timing with two radars and with one. Each single-radar arm is trained and generated using only '
             'that radar\'s candidates and geometry; everything else is identical.\n'
             'Bars: mean over the 5 held-out folds of the subject-wise cross-validation; error bars: sd between folds. '
             'Whole walking course, 40 subjects, 3 seeds, reviewed belt.',fontsize=10)
plt.tight_layout(rect=[0,0,1,0.86]);plt.savefig(f'{M}/bars_dual_timing_rev.png',dpi=140,bbox_inches='tight');plt.close()
print('\n'.join(f'{M}/{n}' for n in ('table_per_subject_rev.png','bars_arms_rev.png','dumbbell_own_vs_perm_rev.png','coverage_scatter_rev.png','bars_dual_timing_rev.png')))
