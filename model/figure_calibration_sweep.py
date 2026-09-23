"""The four headline metrics against calibration length, all arms scored over the same seconds.

Reads calsweep_summary.csv (eval_calibration_sweep.py). Plots the `common` view - every arm scored from
84 s - because in the `natural` view a shorter block also buys a longer and earlier scored region, so the
arms would not be comparable. The `natural` value is drawn as a faint open marker for reference.
The shaded band on the left marks the lengths at which no 42 s few-shot window fits inside the block, so
those arms get the rate anchor and the first window's 6 s belt clamp only - no weight update.
Error bars are between-fold (5 subject-wise CV folds, seed-averaged).

Run: python3 figure_calibration_sweep.py      Env: IN, OUT
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

MTR = '/home/user1/Desktop/UWB_BIOPAC/MTR/rate_recovered'
IN = os.environ.get('IN', f'{MTR}/calsweep_summary.csv')
OUT = os.environ.get('OUT', f'{MTR}/calsweep_rev.png')
S = pd.read_csv(IN)
S = S[S.arm == 'own'] if 'arm' in S else S             # the permuted-radar control lives in calsweep_radar_gap.csv
CO = S[S.view == 'common'].sort_values('cal_s'); NA = S[S.view == 'natural'].sort_values('cal_s')
AO = pd.read_csv(f'{MTR}/calsweep_anchor_only.csv') if os.path.exists(f'{MTR}/calsweep_anchor_only.csv') else None
RED, BLUE, INK, MUTE = '#b3271e', '#2c6fb5', '#1f2733', '#6b7683'
MET = [('rate_err_bpm', 'breathing-rate error  |$\\Delta$bpm|', 'lower is better', True),
       ('CS', 'cosine similarity  CS', 'higher is better', False),
       ('ACF', 'autocorrelation agreement  ACF', 'higher is better', False),
       ('peak_match_pct', 'peak match  (%)', 'higher is better', False)]

fig, axs = plt.subplots(1, 4, figsize=(15.0, 3.7))
for ax, (k, title, dirn, lower) in zip(axs, MET):
    ax.axvspan(0, 42, color='#f0ece4', zorder=0)
    ax.errorbar(CO.cal_s, CO[k], yerr=CO[f'{k}_sd_fold'], color=RED, marker='o', ms=5.5, lw=1.8,
                capsize=3, zorder=3, label='scored from 84 s (comparable)')
    ax.plot(NA.cal_s, NA[k], color=BLUE, marker='o', ms=5.0, lw=0, mfc='none', mew=1.3,
            zorder=2, label='scored from its own block')
    if k == 'rate_err_bpm' and AO is not None:        # what the belt anchor alone predicts, no model
        ax.plot(AO.cal_s, AO.anchor_err, color='#8a6d00', marker='s', ms=4.2, lw=1.4,
                ls=(0, (5, 2)), zorder=2.5, label='calibration anchor alone')
    d = CO[CO.cal_s == 84]
    if len(d):
        ax.axhline(float(d[k].iloc[0]), color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_title(title, fontsize=10.5, weight='bold', color=INK, pad=9)
    ax.set_xlabel('calibration block  (s)', fontsize=9.5)
    ax.text(0.015, 1.012, dirn, transform=ax.transAxes, fontsize=8, color=MUTE, ha='left')
    ax.set_xlim(0, 90); ax.set_xticks([6, 12, 24, 42, 60, 84])
    ax.tick_params(labelsize=8.5)
    ax.grid(axis='y', color='#e2e6ea', lw=0.8, zorder=0)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color('#aab2bb')
for ax in axs:                                    # name the shaded band once per panel, at the bottom
    ax.text(21, ax.get_ylim()[0], ' no few-shot fits\n (anchor only)', ha='center', va='bottom',
            fontsize=7.8, color='#8a7a5c', style='italic')
# CS is the flat one and its axis is necessarily zoomed, so say the span in words
cs = CO.CS
axs[1].text(0.97, 0.06, f'spans only {cs.max()-cs.min():.3f}\nover a 14x longer block',
            transform=axs[1].transAxes, ha='right', va='bottom', fontsize=8.2, color=INK,
            bbox=dict(fc='white', ec='#d6dbe1', lw=0.8, pad=3.2))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=9, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.965))
fig.suptitle('How long does the calibration block have to be?   40 subjects, 3 seeds, '
             'error bars between the 5 CV folds,   reviewed belt',
             fontsize=11.5, weight='bold', y=1.075)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT, dpi=200, bbox_inches='tight'); print(OUT)

# the same thing as a table, for the slide
cols = ['cal_s', 'few_shot_rows', 'scored_s'] + [k for k, _, _, _ in MET]
t = CO[cols + [f'{k}_sd_fold' for k, _, _, _ in MET]].copy()
print('\ncommon view (all arms scored from 84 s)')
print(t.round(3).to_string(index=False))
