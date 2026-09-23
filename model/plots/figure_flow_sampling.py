"""Stage 4 of the model-flow slide, as one tall transparent panel: noise -> waveform -> whole course.

The earlier panel showed only the three denoising snapshots, which says "we ran a diffusion model" and
stops there. The contribution lives in what is injected BETWEEN the steps, so the panel is three rows:
  A  the three snapshots (t = 50, 25, 0)
  B  the per-step loop with its two constraints: the rate band from calibration, and the clamp to the
     previous window's tail (soft RePaint, free for the last 10 steps)
  C  the 42 s windows advancing 36 s at a time with a 6 s raised-cosine crossfade

Run: AGC=9 python3 figure_flow_sampling.py     Env: SUB, SEED, OUT (directory)
"""
import _path  # noqa: F401  (see _path.py)
import os
import numpy as np
os.environ.setdefault('AGC', '9')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from normalize import zn

PRE = '/home/user1/Desktop/UWB_BIOPAC/preprocessed'
MTR = '/home/user1/Desktop/UWB_BIOPAC/MTR/rate_recovered'
OUT = os.environ.get('OUT', f'{MTR}/flow_parts'); os.makedirs(OUT, exist_ok=True)
SUB = int(os.environ.get('SUB', '33')); SEED = int(os.environ.get('SEED', '0'))
FS, NC, CH, W = 17, 14, 51, 714
RED, GRN, YEL, VIO, BLUE = '#b3271e', '#2e7d32', '#c9a227', '#7b4fa8', '#2c6fb5'
INK, MUTE = '#1f2733', '#6b7683'

d = np.load(f'{PRE}/_seam_cont_fc729_K1_soft10_anchor_rc_perm_cnt_g0.2agc_s{SEED}_raw.npz', allow_pickle=True)
out = zn(d[f'sub{SUB:02d}_stream'].astype(float))

fig = plt.figure(figsize=(4.1, 8.4)); fig.patch.set_alpha(0)


def clean(a, ec='#8d99a6'):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(ec); s.set_linewidth(0.8)


# ---- A. three snapshots ------------------------------------------------------
fig.text(0.5, 0.982, 'A.  noise  ->  waveform', ha='center', fontsize=10.5, weight='bold', color=INK)
rng = np.random.RandomState(3); tgt = out[:W]
for i, (lab, sig) in enumerate((('t = 50', rng.randn(W)),
                                ('t = 25', 0.45 * tgt + 0.85 * rng.randn(W)),
                                ('t = 0', tgt))):
    a = fig.add_axes([0.10, 0.895 - 0.072 * i, 0.84, 0.060])
    a.plot(zn(sig), color=RED, lw=0.9); a.set_xlim(0, W); a.set_ylim(-3.4, 3.4); clean(a)
    a.text(0.015, 0.80, lab, transform=a.transAxes, fontsize=8, color=INK, weight='bold')
fig.text(0.52, 0.726, 'one 42 s window,  50 DDIM steps', ha='center', fontsize=8, color=MUTE)

# ---- B. the per-step loop ----------------------------------------------------
ax = fig.add_axes([0.0, 0.325, 1.0, 0.380]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.5, 0.975, 'B.  every step, two constraints', ha='center', fontsize=10.5, weight='bold', color=INK)
HD = 0.056                                        # header band inside a titled box


def box(x, y, w, h, text, fc, ec, fs=7.6, bold=None, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.03',
                                fc=fc, ec=ec, lw=1.0, zorder=2))
    if bold:                                      # title on its own band, body centred BELOW it
        ax.text(x + w / 2, y + h - HD / 2, bold, ha='center', va='center', fontsize=fs + 0.8,
                weight='bold', color=tc, zorder=3)
        ax.text(x + w / 2, y + (h - HD) / 2, text, ha='center', va='center', fontsize=fs,
                color=INK, zorder=3, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
                zorder=3, linespacing=1.45)
    return (x, y, w, h)


def arr(p0, p1, rad=0.0, color='#39424e', lw=1.3, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=10, color=color, lw=lw,
                                 linestyle=ls, zorder=1.6, connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=2, shrinkB=3))


BX, BW = 0.085, 0.74                              # the loop rail lives to the right of the boxes
b0 = box(BX, 0.775, BW, 0.110, r'$\varepsilon_\theta(x_t, t, c)\; \rightarrow\; \hat{x}_0$',
         '#f6ded9', RED, fs=10.5)
b1 = box(BX, 0.535, BW, 0.195, 'pull toward the narrow band at $r$\n$r$ = breath rate from the 84 s\ncalibration   (g = 0.2)',
         '#f7eed2', '#b39320', bold='1.  rate anchor', tc='#8a6d00')
b2 = box(BX, 0.295, BW, 0.195, 'overwrite with the previous\nwindow, then let the last\n10 steps run free',
         '#e9e2f3', VIO, bold='2.  first 6 s  =  already known', tc=VIO)
b3 = box(BX, 0.140, BW, 0.100, r'DDIM step back to $x_{t-1}$', '#e7eef6', '#2c6fb5', fs=9)
for p, q in ((b0, b1), (b1, b2), (b2, b3)):
    arr((p[0] + p[2] / 2, p[1]), (q[0] + q[2] / 2, q[1] + q[3]))
RX, Y0, Y1 = 0.895, 0.190, 0.830                  # square rail: out, up, back in - never behind a box
ax.plot([BX + BW, RX, RX, BX + BW + 0.045], [Y0, Y0, Y1, Y1], color='#8a8f97', lw=1.1,
        solid_joinstyle='round', zorder=1.4)
ax.add_patch(FancyArrowPatch((BX + BW + 0.048, Y1), (BX + BW - 0.002, Y1), arrowstyle='-|>',
                             mutation_scale=10, color='#8a8f97', lw=1.1, zorder=1.4))
ax.text(RX + 0.048, (Y0 + Y1) / 2, '50 x', ha='center', va='center', fontsize=8.5,
        color='#8a8f97', weight='bold', rotation=90)
ax.text(0.5, 0.048, 'nothing here retrains the model', ha='center', va='center', fontsize=7.6,
        style='italic', color='#7a2b23')

# ---- C. windows advancing ----------------------------------------------------
ax = fig.add_axes([0.0, 0.035, 1.0, 0.265]); ax.set_xlim(-2, 132); ax.set_ylim(-0.30, 1.30); ax.axis('off')
ax.text(65, 1.23, 'C.  windows, 36 s apart', ha='center', fontsize=10.5, weight='bold', color=INK,
        transform=ax.transData)
for i, s0 in enumerate((0, 36, 72)):
    yb = 0.78 - 0.24 * i
    ax.add_patch(Rectangle((s0, yb), 42, 0.155, fc='#cfe0f0', ec=BLUE, lw=0.9, zorder=2))
    ax.text(s0 + 24, yb + 0.077, '42 s', ha='center', va='center', fontsize=7.4, color=INK, zorder=4)
    if i:                                          # the first window has no predecessor to clamp to
        ax.add_patch(Rectangle((s0, yb), 6, 0.155, fc=VIO, ec='none', alpha=0.55, zorder=3))
        ax.text(s0 + 3, yb + 0.077, '6', ha='center', va='center', fontsize=6.4, color='white',
                weight='bold', zorder=4)
ax.annotate('', xy=(36, 0.70), xytext=(0, 0.70),
            arrowprops=dict(arrowstyle='<|-|>', color=MUTE, lw=0.9, mutation_scale=8))
ax.text(18, 0.615, 'stride 36 s', ha='center', va='center', fontsize=7.2, color=MUTE)
ax.text(65, 0.10, '6 s raised-cosine crossfade   ->   one continuous waveform\n'
                  'for the whole 5 min course', ha='center', va='center', fontsize=8, color=INK,
        linespacing=1.5)

f = f'{OUT}/part8_sampling_panel.png'
fig.savefig(f, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.05)
plt.close(fig); print(f)
