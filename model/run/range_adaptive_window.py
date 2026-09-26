"""Range-adaptive window, version 2 (author 2026-09-25: keep the range-direction window, fix it; LOS and Parallel alike).

Why v1 (window_extraction_v1.combine) lost the timing: it combined bins with a rank-1 SVD of the RAW
background-subtracted signal, so the weights followed the largest component = walking motion, and no bulk-motion
removal / breathing band-pass was applied before [|z|, dphi].
v2, on the same coarse range map Z the existing LOS uses (data_fc729 Z_r, 5.1 cm bins):
  1 each bin of the window around the path gets the existing LOS chain: I, Q -> rout (regress out bulk amplitude) -> bpf 0.1-0.6 Hz
  2 window per 3 s chunk, range direction only: contiguous bins whose breathing-band energy (chunk mean) stays within
    -3 dB of the path bin, at most +-KW bins (+-15 cm)
  3 combine = phase-align every bin to the path bin on the band-passed signal (complex correlation over the chunk),
    weight by breathing-band energy; if the aligned bins do not agree (|mean coherence| < 0.5) use the path bin alone
  4 [|z|, dphi] per window exactly as the existing channels
Parallel path = body + GT-free offset (energy rule, as parallel_path.py). No belt anywhere.
Outputs (both grids):
  _adv2L8{GTAG}   LOS replaced by v2 LOS, same 8-channel layout (compare with adlos8 = v1)
  _adv2PX12{GTAG} [COM-LOS v2, COM-PAR v2, TV-LOS v2, TV-PAR v2, COM-SR, TV-SR]  (PAIRX=1: LOS/PAR of a radar read jointly)
Run: python3 range_adaptive_window.py"""
import _path  # noqa: F401  (see _path.py)
import os, glob, numpy as np
from multiprocessing import Pool
import signal_processing as RC
from parallel_path import windows, PRE, FS, W
CH = 51; KW = 3


def chain(Zr, bins):
    sig = Zr[np.arange(len(bins)), bins]; amp = np.abs(sig)
    return RC.bpf(RC.rout(np.real(sig), amp)) + 1j * RC.bpf(RC.rout(np.imag(sig), amp))


def adaptive(Zr, path, E, info=None):
    """info: optional list, gets (chunk start, kl, kr, fell_back) per 3 s chunk (for plotting)"""
    T, B = Zr.shape; J = np.arange(-KW, KW + 1)
    S = np.stack([chain(Zr, np.clip(path + j, 0, B - 1)) for j in J], 1)          # (T, 2KW+1) band-passed complex
    Ep = E[np.arange(T)[:, None], np.clip(path[:, None] + J[None, :], 0, B - 1)]   # breathing-band energy on the window
    widths = []; fb = 0; WK = []; ctr = []
    for a in range(0, T, CH):
        b = min(T, a + CH); e = Ep[a:b].mean(0); thr = 0.5 * e[KW]; kl = kr = 0
        while kl < KW and e[KW - kl - 1] >= thr: kl += 1
        while kr < KW and e[KW + kr + 1] >= thr: kr += 1
        sel = np.arange(KW - kl, KW + kr + 1); widths.append(len(sel)); ctr.append((a + b - 1) / 2)
        if info is not None: info.append([a, kl, kr, 0])
        w = np.zeros(2 * KW + 1, complex); w[KW] = 1.0                               # default: path bin alone
        if len(sel) > 1:
            c = S[a:b, KW]; X = S[a:b][:, sel]
            rho = (X * np.conj(c)[:, None]).sum(0) / (np.linalg.norm(X, axis=0) * np.linalg.norm(c) + 1e-12)
            if np.abs(rho[sel != KW]).mean() < 0.5:                                 # bins disagree -> path bin alone
                fb += 1
                if info is not None: info[-1][3] = 1
            else:
                w = np.zeros(2 * KW + 1, complex); w[sel] = e[sel] / e[sel].sum() * np.exp(-1j * np.angle(rho))
        WK.append(w / np.abs(w).sum())
    # weights change smoothly: linear interpolation between chunk centres (no 3 s steps; the path-bin weight is
    # real and positive in every chunk, so all chunks share one phase reference)
    WK = np.array(WK); tt = np.arange(T)
    Wt = np.stack([np.interp(tt, ctr, WK[:, j].real) + 1j * np.interp(tt, ctr, WK[:, j].imag) for j in range(2 * KW + 1)], 1)
    out = (S * Wt).sum(1)
    return out, np.mean(widths), fb / len(widths)


def _subject(u):
    z = np.load(f'{PRE}/sub{u:02d}/data_fc729.npz', allow_pickle=True); g = z['gt_aligned']
    Zc, Zt = z['Z_com'], z['Z_tv']; T = min(len(g), len(Zc), len(Zt)); out = {}; st = {}
    for rad, Zr in (('com', Zc), ('tv', Zt)):
        Zr = Zr[:T]; B = Zr.shape[1]; tr = np.clip(z[f'traj_{rad}'].astype(int)[:T], 0, B - 1); gb = int(np.ravel(z[f'ghost_{rad}'])[0])
        E = RC.bbE(Zr); base = np.median(E, axis=1) + 1e-12; ix = np.arange(T); pr = []
        offs = np.arange(max(1, int(round(0.15 / RC.RES))), int(round(1.60 / RC.RES)) + 1)
        for o in offs:
            p = np.clip(tr + o, 0, B - 1); ok = (tr + o < B) & (np.abs(p - gb) > 3)
            pr.append(np.mean(E[ix[ok], p[ok]] / base[ok]) if ok.sum() > 0.6 * T else 0.0)
        par = np.clip(tr + offs[int(np.argmax(pr))], 0, B - 1)
        for nm, p in (('los', tr), ('par', par)):
            s, wd, fb = adaptive(Zr, p, E); out[f'{rad}_{nm}'] = (np.real(s), np.imag(s)); st[f'{rad}_{nm}'] = (wd, fb)
    return u, out, st


if __name__ == '__main__':
    subs = [int(os.path.basename(os.path.dirname(f))[3:]) for f in sorted(glob.glob(f'{PRE}/sub*/data_fc729.npz'))]
    with Pool(12) as pool: res = pool.map(_subject, subs)
    for k in ('com_los', 'com_par', 'tv_los', 'tv_par'):
        wd = np.mean([r[2][k][0] for r in res]); fb = np.mean([r[2][k][1] for r in res])
        print(f'{k}: mean window {wd:.2f} bins ({wd*RC.RES*100:.0f} cm), fell back to the path bin in {100*fb:.0f} % of chunks')
    for tag, hop_s in (('_fc729', 6.0), ('_fc729_hop1', 1.0)):
        S = np.load(f'{PRE}/_geo_dataset{tag}.npz', allow_pickle=True)['S']; N = len(S)
        base = np.load(f'{PRE}/_iq_candidates{tag}_envdphi.npy').reshape(N, 8, W)
        new = {k: np.zeros((N, 2, W), np.float32) for k in ('com_los', 'com_par', 'tv_los', 'tv_par')}
        for u, o, _ in res:
            rows = np.flatnonzero(S == u)
            for k in new:
                A = windows(*o[k], None, int(hop_s * FS)); m = min(len(rows), len(A)); new[k][rows[:m]] = A[:m]
        for k, c in (('com_los', 0), ('tv_los', 4)):
            r = np.mean([np.corrcoef(new[k][i, 0], base[i, c])[0, 1] for i in range(0, N, 7)])
            print(f'{tag} v2 {k} |z| vs existing single-bin: mean r {r:.3f}')
        np.save(f'{PRE}/_adv2L8{tag}.npy', np.concatenate([new['com_los'], base[:, 2:4], new['tv_los'], base[:, 6:8]], 1))
        np.save(f'{PRE}/_adv2PX12{tag}.npy', np.concatenate([new['com_los'], new['com_par'], new['tv_los'], new['tv_par'], base[:, 2:4], base[:, 6:8]], 1))
        print(f'{tag}: saved _adv2L8 / _adv2PX12', flush=True)
