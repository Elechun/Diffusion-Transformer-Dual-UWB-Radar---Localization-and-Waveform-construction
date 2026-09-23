"""The training loss as a slide-ready equation image (transparent PNG).

Mirrors diffusion_model.py:216-230 exactly: eps-MSE + zone CE + 0.3 * local shape + 0.6 * global shape,
where the two shape terms are 1 - Pearson correlation measured on the x0 reconstructed from the
predicted noise, NOT an MSE, and NOT |rho| (a sign-free loss can be won by flipping the waveform).

  loss_equation.png        the full four-term equation with the forward process and rho spelled out
  loss_equation_1line.png  the one-line version for a crowded slide
Run: python3 figure_loss_equation.py      Env: OUT (directory)
"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.environ.get('OUT', '/home/user1/Desktop/UWB_BIOPAC/MTR/rate_recovered')
os.makedirs(OUT, exist_ok=True)
INK, MUTE = '#1f2733', '#6b7683'
EPS, ZONE, SHL, SHG = '#b3271e', '#2e7d32', '#7b4fa8', '#2c6fb5'
plt.rcParams['mathtext.fontset'] = 'cm'


def save(fig, name):
    f = f'{OUT}/{name}'
    fig.savefig(f, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.06)
    plt.close(fig); print(f)


# ---- full version -------------------------------------------------------------
fig = plt.figure(figsize=(11.6, 4.5)); fig.patch.set_alpha(0)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
T = lambda x, y, s, fs=15, c=INK, ha='center', w='normal': ax.text(
    x, y, s, fontsize=fs, color=c, ha=ha, va='center', weight=w)

T(0.5, 0.955, 'training, one step', 11, MUTE)
T(0.5, 0.855,
  r'$x_0 \in \mathbb{R}^{14\times 51}$  (42 s target),   '
  r'$t \sim \mathcal{U}\{0,\dots,999\}$,   $\varepsilon \sim \mathcal{N}(0, I)$', 13.5)
T(0.5, 0.735,
  r'$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon$'
  r'$\qquad\qquad$'
  r'$\hat{x}_0 = \dfrac{x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta(x_t, t, c)}'
  r'{\sqrt{\bar{\alpha}_t}}$', 16)

# the four terms, laid out on one baseline so each can carry its own colour and caption
y, yc, yl = 0.475, 0.290, 0.208
xs = [0.115, 0.345, 0.625, 0.885]
T(xs[0], y, r'$\mathcal{L} \;=\; \left\Vert \varepsilon_\theta - \varepsilon \right\Vert^2$', 18, EPS)
T(0.225, y, r'$+$', 18)
T(xs[1], y, r'$\dfrac{1}{14}\sum_{k=1}^{14} \mathrm{CE}\!\left(\ell_k,\, z_k\right)$', 18, ZONE)
T(0.475, y, r'$+\;\; 0.3$', 18)
T(xs[2], y, r'$\dfrac{1}{13}\sum_{k=2}^{14}\left(1 - \rho\!\left(\hat{x}_0^{1:k},\, x_0^{1:k}\right)\right)$', 18, SHL)
T(0.775, y, r'$+\;\; 0.6$', 18)
T(xs[3], y, r'$\left(1 - \rho\!\left(\hat{x}_0,\, x_0\right)\right)$', 18, SHG)

for x, c, top, bot in ((xs[0], EPS, 'diffusion', 'predict the noise'),
                       (xs[1], ZONE, 'zone', '14 floors, per 3 s chunk'),
                       (xs[2], SHL, 'shape, local', 'chunk 1..k, cumulative'),
                       (xs[3], SHG, 'shape, global', 'the whole 42 s')):
    ax.plot([x - 0.072, x + 0.072], [yc + 0.036, yc + 0.036], color=c, lw=1.4, alpha=0.55)
    T(x, yc, top, 12.5, c, w='bold'); T(x, yl, bot, 10.5, MUTE)

T(0.185, 0.098,
  r'$\rho(a,b) = \dfrac{\langle a - \bar{a},\; b - \bar{b}\rangle}'
  r'{\Vert a - \bar{a}\Vert \, \Vert b - \bar{b}\Vert}$', 15)
T(0.615, 0.118, 'Pearson, not MSE: invariant to amplitude and offset, so only timing is scored', 12.5)
T(0.615, 0.045, r'signed $\rho$, not $|\rho|$  —  a sign-free loss is won by flipping the waveform',
  11, MUTE)
save(fig, 'loss_equation.png')

# ---- one-line version ---------------------------------------------------------
fig = plt.figure(figsize=(11.2, 1.25)); fig.patch.set_alpha(0)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.5, 0.66,
        r'$\mathcal{L} = \left\Vert \varepsilon_\theta - \varepsilon \right\Vert^2'
        r' \;+\; \mathrm{CE}\!\left(\ell, z\right)'
        r' \;+\; 0.3\,\overline{\left(1 - \rho_{1:k}\right)}'
        r' \;+\; 0.6\left(1 - \rho_{\mathrm{full}}\right)$',
        fontsize=21, color=INK, ha='center', va='center')
ax.text(0.5, 0.17, 'diffusion        zone (14 floors / 3 s)        shape, cumulative        shape, 42 s'
        '        —   $\\rho$ = Pearson, signed',
        fontsize=11, color=MUTE, ha='center', va='center')
save(fig, 'loss_equation_1line.png')
