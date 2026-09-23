"""Is the whole calibration-length curve just the anchor estimate getting better?

For each block length L the deployed anchor is the breath COUNT over the belt's first L seconds, so its
resolution is 60/L bpm - 10 bpm at 6 s, 0.71 bpm at 84 s. This computes, with no model and no GPU, how
far that anchor lands from the subject's own rate over the SCORED region (84 s onward, the `common` view
of eval_calibration_sweep.py), using the same breath-count definition on both sides.

If the model's |dbpm| curve sits on this one, the calibration-length effect is the rate estimate and
nothing else. Where the model is BELOW it, the radar (or the guidance) is recovering something the anchor
alone does not have.

Run: AGC=9 python3 eval_anchor_only.py      Env: CALS, OUT
"""
import _path  # noqa: F401  (see _path.py)
import os, numpy as np
os.environ.setdefault('AGC', '9')
import pandas as pd
from normalize import zn
import metrics_timing as PT
from scipy.signal import find_peaks

PRE = '/home/user1/Desktop/UWB_BIOPAC/preprocessed'; MTR = '/home/user1/Desktop/UWB_BIOPAC/MTR/rate_recovered'
FS = 17; COMMON = 84.0
EXCL = (2, 10, 13, 21, 22)
CALS = [float(c) for c in os.environ.get('CALS', '6,12,18,24,30,42,60,84').split(',')]
OUT = os.environ.get('OUT', f'{MTR}/calsweep_anchor_only.csv')


def count_bpm(x):
    """the deployed rate estimator: breaths counted on the AGC-normalised signal, per minute"""
    p, _ = find_peaks(PT._agc(x, PT.AGC) if PT.AGC > 0 else zn(x), prominence=0.4, distance=int(FS / PT.RHI))
    return len(p) / (len(x) / FS) * 60


subs = sorted(int(d[3:]) for d in os.listdir(PRE)
              if d.startswith('sub') and os.path.isdir(f'{PRE}/{d}')
              and os.path.exists(f'{PRE}/{d}/data_fc729.npz'))
subs = [u for u in subs if u not in EXCL]
rows = []
for u in subs:
    g = np.load(f'{PRE}/sub{u:02d}/data_fc729.npz', allow_pickle=True)['gt_aligned'].astype(float)
    tail = g[int(COMMON * FS):]; tail = tail[np.isfinite(tail)]
    if len(tail) < 60 * FS: continue
    truth = count_bpm(zn(tail))                                   # the subject's rate over the scored region
    pop = None                                                    # filled in below (leave-one-out cohort mean)
    for c in CALS:
        head = g[:int(c * FS)]; head = head[np.isfinite(head)]
        if len(head) < int(c * FS) * 0.9: continue
        rows.append(dict(subject=u, cal_s=c, anchor_bpm=count_bpm(zn(head)), truth_bpm=truth))
A = pd.DataFrame(rows)
A['err'] = (A.anchor_bpm - A.truth_bpm).abs()
# the no-calibration floor: predict every subject with the mean of the OTHER subjects' rates
t = A[A.cal_s == A.cal_s.max()].set_index('subject').truth_bpm
loo = {u: (t.sum() - t[u]) / (len(t) - 1) for u in t.index}
pop_err = float(np.mean([abs(loo[u] - t[u]) for u in t.index]))

S = A.groupby('cal_s').agg(anchor_err=('err', 'mean'), sd=('err', 'std'),
                           n=('err', 'size')).reset_index()
S['resolution_bpm'] = 60.0 / S.cal_s
S.round(4).to_csv(OUT, index=False)
print(f'{len(t)} subjects,  scored region = {COMMON:g} s onward')
print(f'no calibration at all (leave-one-out cohort mean): |dbpm| = {pop_err:.3f}\n')
print(f'{"cal(s)":>7} {"count resolution":>17} {"anchor |dbpm| vs own rate":>27}')
for _, r in S.iterrows():
    print(f'{r.cal_s:>7.0f} {r.resolution_bpm:>16.2f}  {r.anchor_err:>20.3f}')
print(f'\nsaved {OUT}')
