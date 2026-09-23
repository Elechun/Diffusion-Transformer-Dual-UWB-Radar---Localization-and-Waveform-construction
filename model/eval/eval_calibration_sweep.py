"""How long does the calibration block have to be?  The four headline metrics against calibration length.

Every arm is the deployed system with ONE thing changed: the length of the calibration block. What that
block buys changes with its length, and the table has to say so:
  < 42 s   rate anchor (breath count over the block) + the first window's 6 s belt clamp. No weight update -
           a few-shot row IS a 42 s window, so none fits inside a shorter block.
  42 s     the above + few-shot on 1 window
  84 s     the above + few-shot on 8 windows (the deployed setting)

Each arm is scored twice, because the two answer different questions:
  natural   from the end of its own calibration block - what you would actually get in deployment, but
            a short block also means a LONGER and EARLIER scored region, so the arms are not comparable
  common    from 84 s for every arm - the same subjects, the same seconds, the same number of breaths.
            This is the comparison; `natural` is the deployment number.
The belt reference is the human-reviewed one (belt_peaks_reviewed.mat); CS and ACF do not use the marks.
Error bars are between-fold (the 5 subject-wise CV folds), not between-subject.

Run: AGC=9 python3 eval_calibration_sweep.py      Env: CALS (list), OUT
"""
import _path  # noqa: F401  (see _path.py)
import os, numpy as np
os.environ.setdefault('AGC', '9')
from scipy.io import loadmat
from scipy.signal import find_peaks
from normalize import zn
import metrics_windowed as T
import metrics_timing as PT

PRE = '/home/user1/Desktop/UWB_BIOPAC/preprocessed'; MTR = '/home/user1/Desktop/UWB_BIOPAC/MTR/rate_recovered'
FS = 17; COMMON = 84.0                       # every arm is also scored from this second, so the arms compare
        # a CALS token may carry a variant suffix: "84:nofs" is the 84 s block with few-shot forced off
CALS = [(float(c.split(':')[0]), c.split(':')[1] if ':' in c else '')
        for c in os.environ.get('CALS', '6,12,18,24,30,42,60,84,84:nofs').split(',')]
OUT = os.environ.get('OUT', MTR); os.makedirs(OUT, exist_ok=True)
R = {int(r.sub): r for r in np.atleast_1d(loadmat(
    '/home/user1/Desktop/UWB_BIOPAC/MTR/belt_peak_review/belt_peaks_reviewed.mat',
    squeeze_me=True, struct_as_record=False)['R'])}
BASE = '_seam_cont_fc729_K1_soft10_anchor_rc_perm'
MET = [('rate_err_bpm', '|Δbpm|'), ('CS', 'CS'), ('ACF', 'ACF'), ('peak_match_pct', 'peak match %')]
KEYS = [k for k, _ in MET]


def tag(c, var=''):
    base = f'{BASE}_cnt_g0.2agc' if c == 84 else f'{BASE}_cal{c:g}_cnt_g0.2agc'
    return f'{base}_{var}' if var else base


def mpick(x):
    p, _ = find_peaks(PT._agc(x, PT.AGC) if PT.AGC > 0 else zn(x), prominence=0.4, distance=int(FS / PT.RHI))
    return p.astype(float)


def match(a, b, tol):
    used = set(); hit = []
    for ai in a:
        c = [(abs(ai - bj), j) for j, bj in enumerate(b) if j not in used and abs(ai - bj) <= tol]
        if c: dd, j = min(c); used.add(j); hit.append(dd)
    return hit


def score(m, b, u, off):
    """off = sample index into the whole recording at which this (already cut) stream starts."""
    m, b = zn(m), zn(b); n = min(len(m), len(b)); m, b = m[:n], b[:n]
    dur = n / FS
    # belt marks are stored as 1-based sample indices into the whole recording
    pb = np.atleast_1d(np.asarray(R[u].peaks, float)) - 1.0 - off
    vb = np.atleast_1d(np.asarray(R[u].valleys, float)) - 1.0 - off
    pb = pb[(pb >= 0) & (pb < n)]; vb = vb[(vb >= 0) & (vb < n)]
    if len(pb) < 3 or len(vb) < 3: return None
    pm = mpick(m)
    hp = match(pb, pm, 0.5 * np.median(np.diff(pb))); mm = T.metrics(m, b)
    return dict(rate_err_bpm=abs(len(pm) - len(pb)) / dur * 60, CS=mm['CSp'], ACF=mm['ACF'],
                peak_match_pct=100 * len(hp) / len(pb))


rows = []
for c, var in CALS:
    for s in (0, 1, 2):
        f = f'{PRE}/{tag(c, var)}_s{s}_raw.npz'
        if not os.path.exists(f): print(f'  missing {os.path.basename(f)}'); continue
        d = np.load(f, allow_pickle=True)
        for u in [int(x) for x in d['subs']]:
            b = np.asarray(d[f'sub{u:02d}_bstream'], float).reshape(-1)
            c0 = int(round(c * FS))                       # this stream's first sample in the recording
            for arm, key in (('own', 'stream'), ('perm', 'stream_perm')):   # perm = the radar control: same
                k = f'sub{u:02d}_{key}'                                     # chain, conditioning radar taken
                if k not in d.files or len(d[k]) == 0: continue             # from a window >=72 s away
                m = np.asarray(d[k], float).reshape(-1)
                for view, cut in (('natural', 0), ('common', int(round(COMMON * FS)) - c0)):
                    if cut < 0 or cut >= min(len(m), len(b)): continue
                    sc = score(m[cut:], b[cut:], u, c0 + cut)
                    if sc: rows.append(dict(cal_s=c, variant=var, seed=s, subject=u, view=view, arm=arm,
                                            scored_s=(min(len(m), len(b)) - cut) / FS, **sc))
    n = sum(1 for r in rows if r['cal_s'] == c and r['variant'] == var
            and r['view'] == 'common' and r['arm'] == 'own')
    print(f'  CALS {c:g}{" " + var if var else ""}: {n} common rows', flush=True)

import pandas as pd
from scipy.stats import wilcoxon
SU = pd.DataFrame(rows)
subs = np.array(sorted(SU.subject.unique()))
pmx = np.random.RandomState(0).permutation(len(subs))
FOLD = {int(u): i for i in range(5) for u in subs[pmx[i::5]]}
SU['fold'] = SU.subject.map(FOLD)
SF = SU.groupby(['view', 'arm', 'cal_s', 'variant', 'seed', 'fold'])[KEYS].mean().reset_index()   # the CV unit

summ = []
for (view, arm, c, var), g in SF.groupby(['view', 'arm', 'cal_s', 'variant']):
    sel = (SU.view == view) & (SU.arm == arm) & (SU.cal_s == c) & (SU.variant == var)
    d = dict(view=view, arm=arm, cal_s=c, variant=var, n_subjects=SU[sel].subject.nunique(),
             scored_s=SU[sel].scored_s.mean(),
             few_shot_rows=0 if var == 'nofs' else max(0, int((c - 42.0) // 6) + 1))
    for k in KEYS:
        fm = g.groupby('fold')[k].mean()
        d[f'{k}'] = fm.mean(); d[f'{k}_sd_fold'] = fm.std(ddof=1)
    summ.append(d)
S = pd.DataFrame(summ).sort_values(['view', 'arm', 'cal_s', 'variant'])

# own vs perm, paired over subjects (seed-averaged) - does the radar still matter at short blocks?
gap = []
for c, var in sorted({(r['cal_s'], r['variant']) for r in rows}):
    p = (SU[(SU.view == 'common') & (SU.cal_s == c) & (SU.variant == var)]
         .groupby(['arm', 'subject'])[KEYS].mean().unstack(0))
    if ('CS', 'perm') not in p: continue
    row = dict(cal_s=c, variant=var)
    for k in KEYS:
        o, q = p[(k, 'own')], p[(k, 'perm')]
        ok = o.notna() & q.notna()
        row[f'{k}_own_minus_perm'] = float((o[ok] - q[ok]).mean())
        row[f'{k}_p'] = float(wilcoxon(o[ok], q[ok]).pvalue) if ok.sum() >= 6 else np.nan
    gap.append(row)
GAP = pd.DataFrame(gap)
GAP.round(4).to_csv(f'{OUT}/calsweep_radar_gap.csv', index=False)
S.round(4).to_csv(f'{OUT}/calsweep_summary.csv', index=False)
SU.round(4).to_csv(f'{OUT}/calsweep_by_subject.csv', index=False)
SF.round(4).to_csv(f'{OUT}/calsweep_by_seed_fold.csv', index=False)

for view in ('common', 'natural'):
    v = S[(S.view == view) & (S.arm == 'own')]
    if not len(v): continue
    print(f'\n=== {view} ===' + ('   (all arms scored from 84 s - THE comparison)' if view == 'common'
                                 else '   (each arm from its own CALS; scored region differs - deployment number)'))
    print(f'{"cal(s)":>10} {"fewshot":>8} {"scored(s)":>10} ' +
          ' '.join(f'{lab:>16}' for _, lab in MET))
    for _, r in v.iterrows():
        cells = ' '.join(f'{r[k]:>9.3f}±{r[k+"_sd_fold"]:.3f}' for k in KEYS)
        name = f'{r.cal_s:.0f}' + (f' {r.variant}' if r.variant else '')
        print(f'{name:>10} {r.few_shot_rows:>8d} {r.scored_s:>10.0f} {cells}')

print('\n=== radar control: own - permuted radar, paired over subjects, common view ===')
print(f'{"cal(s)":>10} ' + ' '.join(f'{lab:>22}' for _, lab in MET))
for _, r in GAP.iterrows():
    name = f'{r.cal_s:.0f}' + (f' {r.variant}' if r.variant else '')
    print(f'{name:>10} ' + ' '.join(f'{r[k+"_own_minus_perm"]:>+13.3f} p={r[k+"_p"]:.3f}' for k in KEYS))
print(f'\nsaved {OUT}/calsweep_summary.csv, calsweep_by_subject.csv, calsweep_by_seed_fold.csv, calsweep_radar_gap.csv')
