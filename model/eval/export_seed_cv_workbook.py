"""Every number behind the comparison figures, broken out by seed and by cross-validation fold, as one workbook.

Sheets
  README          what each sheet holds, how the folds are defined, and which belt reference is used
  by_seed_fold    one row per (arm, seed, fold): the fold's held-out subjects scored with that seed's weights
  by_fold         seed-averaged per fold  (mean over seeds of the fold means)  + sd across seeds
  by_seed         fold-averaged per seed  (mean over the 5 folds)             + sd across folds
  summary         per arm: mean over folds, sd between folds, sd between seeds, sd between subjects, SE, 95% CI
  by_subject      one row per (arm, subject): mean over seeds, and the seed-to-seed sd for that subject
  radar_control   the same breakdown for the radar-permuted control of each arm

Folds are the subject-wise split used in training (numpy RandomState(0).permutation, 5 groups), so a row's
"fold" is the group of 8 subjects that were held out of the model that scored them.
Run: AGC=9 python3 export_seed_cv_workbook.py      Env: OUT
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
FS = 17; CALN = int(84 * FS)
OUT = os.environ.get('OUT', f'{MTR}/seed_cv_results.xlsx')
R = {int(r.sub): r for r in np.atleast_1d(loadmat(
    '/home/user1/Desktop/UWB_BIOPAC/MTR/belt_peak_review/belt_peaks_reviewed.mat',
    squeeze_me=True, struct_as_record=False)['R'])}
D = '_seam_cont_fc729_K1_soft10_anchor_rc'
ARMS = [('DiT (ours)',                 f'{D}_perm_cnt_g0.2agc'),
        ('DiT, raw I/Q + rotation',    f'{D}_iniq_candidates_rot_perm_cnt_iniq_candidates_g0.2agc'),
        ('TCN + adapt + anchor',       '_baseline_whole_tcn_iq+envdphi_fs_anc'),
        ('U-Net + adapt + anchor',     '_baseline_whole_unet_iq+envdphi_fs_anc'),
        ('Transformer + adapt + anchor', '_baseline_whole_xf_iq+envdphi_fs_anc'),
        ('IQ-VED + adapt + anchor',    '_baseline_whole_ved_iq+envdphi_fs_anc'),
        ('RF-Carer (their SP + net)',  '_baseline_whole_rfcarer_rfcsp_fs_anc')]
MET = [('rate_err_bpm', '|Δbpm|, breath count'), ('CS', 'cosine similarity'), ('ACF', 'autocorrelation agreement'),
       ('peak_match_pct', 'peaks matched within half a period'), ('valley_match_pct', 'valleys matched within half a period'),
       ('time_err_s', 'mean |Δt| over matched peaks and valleys'), ('match_05s_pct', 'peaks matched within ±0.5 s')]
KEYS = [k for k, _ in MET]


def mpick(x):
    p, _ = find_peaks(PT._agc(x, PT.AGC) if PT.AGC > 0 else zn(x), prominence=0.4, distance=int(FS / PT.RHI))
    return p.astype(float)


def marks(u, n, which):
    v = np.atleast_1d(np.asarray(getattr(R[u], which), float)) - 1.0 - CALN
    return v[(v >= 0) & (v < n)]


def match(a, b, tol):
    used = set(); hit = []
    for ai in a:
        c = [(abs(ai - bj), j) for j, bj in enumerate(b) if j not in used and abs(ai - bj) <= tol]
        if c: d, j = min(c); used.add(j); hit.append(d)
    return hit


def score(m, b, u):
    m, b = zn(m), zn(b); n = min(len(m), len(b)); m, b = m[:n], b[:n]; dur = n / FS
    pb, vb = marks(u, n, 'peaks'), marks(u, n, 'valleys'); pm, vm = mpick(m), mpick(-m)
    if len(pb) < 3 or len(vb) < 3: return None
    hp = match(pb, pm, 0.5 * np.median(np.diff(pb))); hv = match(vb, vm, 0.5 * np.median(np.diff(vb)))
    h5 = match(pb, pm, 0.5 * FS); mm = T.metrics(m, b)
    dtp = np.mean(hp) / FS if hp else np.nan; dtv = np.mean(hv) / FS if hv else np.nan
    return dict(rate_err_bpm=abs(len(pm) - len(pb)) / dur * 60, CS=mm['CSp'], ACF=mm['ACF'],
                peak_match_pct=100 * len(hp) / len(pb), valley_match_pct=100 * len(hv) / len(vb),
                time_err_s=np.nanmean([dtp, dtv]), match_05s_pct=100 * len(h5) / len(pb))


# ---- collect: arm -> which -> (seed, subject) -> metrics ----
raw = {}
for lab, tag in ARMS:
    for s in (0, 1, 2):
        f = f'{PRE}/{tag}_s{s}_raw.npz'
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True)
        for u in [int(x) for x in d['subs']]:
            for which, key in (('own', 'stream'), ('perm', 'stream_perm')):
                k = f'sub{u:02d}_{key}'
                if k not in d.files or len(d[k]) == 0: continue
                sc = score(d[k], d[f'sub{u:02d}_bstream'], u)
                if sc: raw.setdefault((lab, which), {})[(s, u)] = sc
    print(f'  {lab}: {len(raw.get((lab,"own"),{}))} (seed, subject) own rows', flush=True)

subs = np.array(sorted({u for v in raw.values() for (_, u) in v}))
pmx = np.random.RandomState(0).permutation(len(subs))
FOLD = {int(u): i for i in range(5) for u in subs[pmx[i::5]]}

import pandas as pd
rows_sf = []; rows_su = []
for (lab, which), v in raw.items():
    for (s, u), sc in v.items():
        rows_su.append(dict(arm=lab, which=which, seed=s, fold=FOLD[u], subject=f'sub{u:02d}', **sc))
SU = pd.DataFrame(rows_su)
SF = (SU.groupby(['arm', 'which', 'seed', 'fold'])[KEYS].mean()
        .join(SU.groupby(['arm', 'which', 'seed', 'fold']).size().rename('n_subjects')).reset_index())
BF = (SF.groupby(['arm', 'which', 'fold'])[KEYS].agg(['mean', 'std'])
        .pipe(lambda d: d.set_axis([f'{a}_{b}' for a, b in d.columns], axis=1)).reset_index())
BS = (SF.groupby(['arm', 'which', 'seed'])[KEYS].agg(['mean', 'std'])
        .pipe(lambda d: d.set_axis([f'{a}_{b}' for a, b in d.columns], axis=1)).reset_index())
SUBJ = (SU.groupby(['arm', 'which', 'subject'])[KEYS].agg(['mean', 'std'])
          .pipe(lambda d: d.set_axis([f'{a}_{b}' for a, b in d.columns], axis=1)).reset_index())

summ = []
for (lab, which), v in raw.items():
    d = dict(arm=lab, which=which)
    sf = SF[(SF.arm == lab) & (SF.which == which)]
    persub = SU[(SU.arm == lab) & (SU.which == which)].groupby('subject')[KEYS].mean()
    for k in KEYS:
        fold_means = sf.groupby('fold')[k].mean(); seed_means = sf.groupby('seed')[k].mean()
        sub_v = persub[k].dropna().values
        d[f'{k}__mean'] = fold_means.mean()
        d[f'{k}__sd_between_folds'] = fold_means.std(ddof=1)
        d[f'{k}__sd_between_seeds'] = seed_means.std(ddof=1)
        d[f'{k}__sd_between_subjects'] = sub_v.std(ddof=1)
        d[f'{k}__se_over_subjects'] = sub_v.std(ddof=1) / np.sqrt(len(sub_v))
        d[f'{k}__ci95_half_width'] = 1.96 * sub_v.std(ddof=1) / np.sqrt(len(sub_v))
        d[f'{k}__n_subjects'] = len(sub_v)
    summ.append(d)
SUM = pd.DataFrame(summ)

readme = pd.DataFrame({'': [
    'Whole walking course after the 84 s calibration block. Reference: the human-reviewed BIOPAC peaks',
    '(belt_peaks_reviewed.mat, +22 marks over 16 subjects); CS and ACF do not depend on the peak marks.',
    '',
    'Cross-validation: subject-wise, 5 folds, numpy RandomState(0).permutation over the 40-subject cohort',
    '(2, 10, 13, 21, 22 excluded). A row\'s "fold" is the group of 8 subjects held out of the model that scored them.',
    'Seeds 0, 1, 2 are independent trainings of all 5 folds, so there are 15 trained models per arm.',
    '',
    'SHEETS',
    'by_seed_fold   one row per (arm, seed, fold) - the smallest unit that exists; everything else aggregates this',
    'by_fold        mean and sd ACROSS SEEDS for each fold',
    'by_seed        mean and sd ACROSS FOLDS for each seed',
    'summary        per arm: mean over folds, and four different spreads, see below',
    'by_subject     one row per (arm, subject): mean over seeds and the seed-to-seed sd for that subject',
    'radar_control  the same sheets restricted to which = perm',
    '',
    'WHICH SPREAD TO QUOTE',
    'sd_between_folds    how much the estimate moves with which 8 subjects were held out (CV convention)',
    'sd_between_seeds    how much it moves with the training seed - small here, the training is stable',
    'sd_between_subjects how much subjects differ from each other; a property of the cohort, NOT of the model.',
    '                    This is the large number that made the earlier error bars look alarming.',
    'se_over_subjects    sd_between_subjects / sqrt(n) - the uncertainty of the cohort mean',
    'ci95_half_width     1.96 x se_over_subjects',
    '',
    'METRICS',
] + [f'{k:18s} {t}' for k, t in MET] + [
    '',
    'NOTE ON THE RF-Carer ROW: its published signal-process layer and network are run on our recordings with our',
    'training, so these are not that paper\'s reported numbers. Its ground-truth-driven sign step was removed.',
]})

with pd.ExcelWriter(OUT, engine='openpyxl') as xl:
    readme.to_excel(xl, 'README', index=False, header=False)
    for name, df in (('by_seed_fold', SF), ('by_fold', BF), ('by_seed', BS), ('summary', SUM), ('by_subject', SUBJ)):
        d = df[df.which == 'own'].drop(columns='which') if 'which' in df else df
        d.round(4).to_excel(xl, name, index=False)
    pd.concat([SF[SF.which == 'perm'], ]).round(4).to_excel(xl, 'radar_control', index=False)
    for ws in xl.book.worksheets:
        ws.freeze_panes = 'A2'
        for col in ws.columns:
            w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max(w + 2, 10), 34)
for name, df in (('by_seed_fold', SF), ('by_subject', SUBJ), ('summary', SUM)):
    df.round(4).to_csv(f'{MTR}/seed_cv_{name}.csv', index=False)
print(f'\nsaved {OUT}')
print(f'  by_seed_fold {len(SF)} rows ({SF.arm.nunique()} arms x 3 seeds x 5 folds x own/perm)')
print(SUM[SUM.which == 'own'][['arm', 'rate_err_bpm__mean', 'rate_err_bpm__sd_between_folds',
                               'rate_err_bpm__sd_between_seeds', 'rate_err_bpm__sd_between_subjects']].round(3).to_string(index=False))
