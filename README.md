# Dual-radar UWB respiration: signal processing and diffusion reconstruction

Respiration monitoring from **two IR-UWB radars while the subject walks a course** (Novelda X4M03,
7.29 GHz, 17 frames/s, COM and TV units at ~90°). The repository holds two stages of the same project:

| | |
|---|---|
| **[`model/`](model/)** | the deployed system — a conditional diffusion transformer that reconstructs the respiratory waveform and a 14-zone floor position every 3 s. **Start here.** |
| `model/signal_processing.py` | the signal-processing front end, built on the RF-Carer (SenSys) method: baseband, background removal, per-radar body and static-reflector tracking. |

See **[model/README.md](model/README.md)** for the deployed model, its configuration and its results, and
**[FINDINGS.md](FINDINGS.md)** for the earlier signal-processing stage that preceded it.

## Data notice

**No recordings and no trained weights are in this repository, and none may be added.**
The BIOPAC belt signals are human physiological recordings and the dataset is access-restricted.
`.gitignore` blocks `*.npz`, `*.npy`, `*.mat`, `*.csv`, `*.acq`, `*.dat` and per-subject figures;
scripts read data through a local, git-ignored `data/` (or `preprocessed/`) folder.

Figures published here are method diagrams and cohort-level results only. Plots of an individual
subject's waveform are not published, since a printed trace can be digitised back into the recording.

## The deployed system in one paragraph

Each radar gives two candidate range bins — the tracked body (LOS) and a static reflector — and each
is reduced to magnitude and phase difference, so absolute phase is never used. Eight such channels over
a 42 s window become 14 chunk tokens; a DiT with cross-chunk attention denoises a respiratory waveform
from noise in 50 DDIM steps while two constraints are applied at every step: a pull toward the
breathing rate counted from an 84 s calibration block, and a clamp of the first 6 s to the previous
window. Windows advance 36 s with a raised-cosine crossfade, giving one continuous trace for the whole
course. The same forward pass emits the floor zone for each 3 s chunk.

40 subjects, 3 seeds, subject-wise 5-fold cross-validation, human-reviewed belt reference:
**breathing-rate error 0.954 bpm · cosine similarity 0.892 · ACF 0.863 · peak match 90.8 % · zone accuracy 0.929.**

## Signal-processing front end (`model/signal_processing.py`)

Raw UWB → baseband → **Matrix X** (signal compensation) → **Y** (temporal background, EMA) →
**Z** (range background, DBSCAN) → influenced-bin selection → extraction.
Two candidates per radar: **LOS** = migrating body (`body_traj_stab`, mobility-CV Viterbi + Kalman/RTS)
and the **static reflector** (`ghost_static`).

## Credit

The signal-processing front end follows the RF-Carer (SenSys) method, and that method is also run on
our recordings as a comparison arm. It is cited wherever it is used.
