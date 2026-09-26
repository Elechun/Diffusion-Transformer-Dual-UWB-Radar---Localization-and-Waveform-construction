"""Parallel path extracted EXACTLY like the deployed LOS channel (author 2026-09-25: keep SR, combine Parallel well).
Deployed LOS = Z_r[t, traj(t)] (coarse bins, data_fc729), I and Q each rout (bulk-motion regression) + bpf 0.1-0.6 Hz,
z-scored per window, then [|z|, dphi] z-scored per window.
Parallel = the same chain on Z_r[t, traj(t) + o], o chosen per subject/radar by breathing-band energy (paraghost rule,
0.15-1.60 m behind the body, away from the static SR bin). No belt.
Check: the LOS rebuilt by this code must equal deployed channels 0,1,4,5.
Outputs (both grids):
  _parS12{GTAG}.npy  deployed 8 + [COM-PAR, TV-PAR]                         (plain concat)
  _parX12{GTAG}.npy  [COM-LOS, COM-PAR, TV-LOS, TV-PAR, COM-SR, TV-SR]      (for PAIRX=1: LOS and PAR of one radar read jointly)
Run: python3 parallel_path.py"""
import _path  # noqa: F401  (see _path.py)
import os, glob, numpy as np
from multiprocessing import Pool
import signal_processing as RC
PRE = '/home/user1/Desktop/UWB_BIOPAC/preprocessed'; FS = 17; NC, CH = 14, 51; W = NC * CH
zn = lambda v: (v - v.mean(-1, keepdims=True)) / (v.std(-1, keepdims=True) + 1e-9)


def iq(Zr, bins, amp_ok=True):
    sig = Zr[np.arange(len(bins)), bins]; amp = np.abs(sig)
    return RC.bpf(RC.rout(np.real(sig), amp)), RC.bpf(RC.rout(np.imag(sig), amp))


def _subject(u):
    z = np.load(f'{PRE}/sub{u:02d}/data_fc729.npz', allow_pickle=True); g = z['gt_aligned']
    Zc, Zt = z['Z_com'], z['Z_tv']; T = min(len(g), len(Zc), len(Zt)); out = {}; offs_m = {}
    for rad, Zr in (('com', Zc), ('tv', Zt)):
        Zr = Zr[:T]; B = Zr.shape[1]; tr = np.clip(z[f'traj_{rad}'].astype(int)[:T], 0, B - 1); gb = int(np.ravel(z[f'ghost_{rad}'])[0])
        E = RC.bbE(Zr); base = np.median(E, axis=1) + 1e-12; ix = np.arange(T); pr = []
        offs = np.arange(max(1, int(round(0.15 / RC.RES))), int(round(1.60 / RC.RES)) + 1)
        for o in offs:
            p = np.clip(tr + o, 0, B - 1); ok = (tr + o < B) & (np.abs(p - gb) > 3)
            pr.append(np.mean(E[ix[ok], p[ok]] / base[ok]) if ok.sum() > 0.6 * T else 0.0)
        o = offs[int(np.argmax(pr))]; offs_m[rad] = o * RC.RES
        out[rad + '_los'] = iq(Zr, tr); out[rad + '_par'] = iq(Zr, np.clip(tr + o, 0, B - 1))
    return u, out, offs_m


def windows(I, Q, S_u, hop):
    n = (len(I) - W) // hop + 1; rows = []
    for j in range(n):
        s0 = j * hop; zc = zn(I[s0:s0 + W]) + 1j * zn(Q[s0:s0 + W])
        dph = np.angle(zc[1:] * np.conj(zc[:-1])); dph = np.concatenate([dph[:1], dph])
        rows.append(np.stack([zn(np.abs(zc)), zn(dph)]))
    return np.array(rows, np.float32)


if __name__ == '__main__':
    subs = [int(os.path.basename(os.path.dirname(f))[3:]) for f in sorted(glob.glob(f'{PRE}/sub*/data_fc729.npz'))]
    with Pool(12) as pool: res = pool.map(_subject, subs)
    print('Parallel offset (m) median COM %.2f TV %.2f' % tuple(np.median([r[2][k] for r in res]) for k in ('com', 'tv')))
    for tag, hop_s in (('_fc729', 6.0), ('_fc729_hop1', 1.0)):
        S = np.load(f'{PRE}/_geo_dataset{tag}.npz', allow_pickle=True)['S']; N = len(S)
        base = np.load(f'{PRE}/_iq_candidates{tag}_envdphi.npy').reshape(N, 8, W)
        new = {k: np.zeros((N, 2, W), np.float32) for k in ('com_los', 'com_par', 'tv_los', 'tv_par')}
        for u, o, _ in res:
            rows = np.flatnonzero(S == u)
            for k in new:
                A = windows(*o[k], None, int(hop_s * FS)); m = min(len(rows), len(A)); new[k][rows[:m]] = A[:m]
        for k, c in (('com_los', 0), ('tv_los', 4)):
            e = np.abs(new[k][:, 0] - base[:, c]).max(); print(f'{tag} rebuilt {k} vs deployed ch{c}: max |diff| {e:.2e}')
        np.save(f'{PRE}/_parS12{tag}.npy', np.concatenate([base, new['com_par'], new['tv_par']], 1))
        np.save(f'{PRE}/_parX12{tag}.npy', np.concatenate([base[:, 0:2], new['com_par'], base[:, 4:6], new['tv_par'], base[:, 2:4], base[:, 6:8]], 1))
        print(f'{tag}: saved _parS12 / _parX12 ({N} rows)', flush=True)
