# Dual-radar diffusion model for walking respiration

Reconstructs a continuous respiratory waveform, and a 14-zone floor position every 3 s, from two
IR-UWB radars (Novelda X4M03, 7.29 GHz, 17 frames/s) while the subject walks a course.

**No data and no trained weights are published here.** The BIOPAC belt recordings are human
physiological data and are access-restricted; `.gitignore` blocks `*.npz`, `*.npy`, `*.mat`, `*.csv`.
The code expects them under a local `preprocessed/` directory.

## Deployed configuration

```
AGC=9 MODE=soft RFREE=10 ANCHOR=1 WTAG=_rc PERM=1 CALEST=count GUIDE=0.2 CALS=84
python3 generate_stream.py
```

| | |
|---|---|
| input | 8 channels = COM/TV radar x LOS/static-reflector x (\|z\|, phase difference) |
| window | 42 s = 14 chunks x 51 samples; windows advance 36 s with a 6 s crossfade |
| network | DiT, 4 blocks, d_model 192, 4 heads, AdaLN on the diffusion timestep, conditions added |
| sampling | DDIM 50 steps, classifier-free guidance w=3 against a radar-zeroed reference |
| calibration | first 84 s: breath-count rate anchor + few-shot on 8 windows (100 steps, lr 5e-5) |
| loss | eps-MSE + zone cross-entropy + 0.3 x cumulative-chunk shape + 0.6 x whole-window shape, shape = 1 - Pearson |

## Modules

| file | role |
|---|---|
| `signal_processing.py` | baseband, range compensation, clutter removal, per-radar body / static-reflector tracking |
| `conditioner.py` | per-chunk encoder producing the 14 tokens |
| `br_candidates.py` | breathing-rate-compensated candidate channels |
| `augment.py` | random-crop augmentation used by the deployed weights (`WTAG=_rc`) |
| `network.py` | the DiT itself, rotation helper, CFG scale and amplitude rescale |
| `diffusion_model.py` | dataset assembly, forward process, training loop |
| `loss.py` | the shape terms (cumulative-chunk and whole-window) |
| `rate_condition.py` | rate as a conditioning channel (studied, not deployed) |
| `generate_stream.py` | **deployed inference**: continuation sampling over the whole course |
| `rate_guidance.py` | per-step narrow-band pull toward the calibration rate (g=0.2, sigma=1.2 bpm) |
| `fewshot.py` | few-shot adaptation on the calibration block; `DONOR=1` runs the donor control |
| `seam.py` | window joining and the seam ratio |
| `determinism.py` | fixed noise and seeding |
| `metrics.py` `metrics_windowed.py` `metrics_timing.py` `tables.py` | scoring |
| `gt_audit.py` | which ground truth a run was trained and scored against |
| `eval_calibration_sweep.py` `eval_anchor_only.py` | calibration-length study |
| `figure_*.py` `export_seed_cv_workbook.py` | figures and the per-seed / per-fold workbook |

## Results

40 subjects, 3 seeds, subject-wise 5-fold CV, human-reviewed belt reference, error bars between folds.

Deployed: **\|dbpm\| 0.954 · CS 0.892 · ACF 0.863 · peak match 90.81 %** (zone accuracy 0.929).

### Calibration length (all arms scored over the same seconds)

| block (s) | few-shot rows | \|dbpm\| | CS | ACF | peak match % |
|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 2.189 | 0.884 | 0.638 | 87.94 |
| 12 | 0 | 1.689 | 0.887 | 0.715 | 89.01 |
| 24 | 0 | 1.414 | 0.889 | 0.775 | 89.61 |
| 30 | 0 | 1.168 | 0.890 | 0.825 | 90.64 |
| 42 | 1 | 1.120 | 0.889 | 0.834 | 90.38 |
| 60 | 4 | 1.068 | 0.894 | 0.859 | 90.95 |
| **84** | 8 | **0.957** | 0.892 | 0.863 | 90.82 |

CS moves 0.010 across a 14x change in block length — the size of one fold error bar. The calibration
block buys breathing rate and periodicity; it does not buy waveform shape.

### What the calibration block buys, separated

| arm | \|dbpm\| | CS | ACF | peak match % |
|---|---:|---:|---:|---:|
| own belt, 8 rows (deployed) | 0.957 | 0.892 | 0.863 | 90.82 |
| no few-shot | 1.229 | 0.888 | 0.845 | 89.31 |
| another subject's belt, 8 rows | 1.472 | 0.884 | 0.832 | 89.03 |

Paired Wilcoxon over 40 subjects: own vs donor \|dbpm\| -0.516 (p=0.0007), CS +0.008 (p=0.0094);
donor vs none \|dbpm\| +0.243 (p=0.016), CS -0.005 (p=0.038). Adapting on somebody else's belt is
worse than not adapting at all, so the gain is personalisation and not the extra gradient steps.

Past ~30 s the anchor itself adds nothing: with few-shot held off, lengthening 30 s -> 84 s does not
improve \|dbpm\| (1.168 -> 1.229, n.s.).

## Notes

- The `RF-Carer` label in `export_seed_cv_workbook.py` names the comparison arm that runs that paper's
  published signal processing and network on our recordings, and is meant to stay.
- Figures under `figures/` are method and aggregate-result plots only. Per-subject waveform plots are
  not published, since they are recordings of individual people.
