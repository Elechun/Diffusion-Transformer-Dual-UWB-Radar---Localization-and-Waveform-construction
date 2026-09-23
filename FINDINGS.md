# Findings — RF-Carer UWB Respiration SP on Continuous-Walking Dual-Radar Data

> **Note.** This is the write-up of the signal-processing stage that preceded the model in `model/`.
> The figures and scripts it was written against are not published: they plot individual subjects'
> BIOPAC belt traces at a resolution from which the recording can be digitised, and the dataset is
> access-restricted. The findings below stand on their own.


## 1. What works (achievements)

**Restored RF-Carer SP pipeline.** Correct carrier (RATIO=0.375, FC=8.748 GHz) + dual-domain background
removal (temporal EMA + range DBSCAN) → clean **Matrix X → Y → Z**. This exceeds the prior Spikformer
signal ceiling.

**Influenced-bin as a body-trajectory problem (reframed).** In walking there is no fixed LOS bin — the
body's range at each instant *is* the LOS. We track it with a mobility-CV Viterbi tracker + Kalman/RTS
smoothing (`body_traj_stab`), which follows the full back-and-forth walk cleanly and cuts trajectory jerk
~14× while preserving true motion. **Ghost is redefined** as the strongest *static* reflector off the
trajectory (`ghost_static`), replacing the arbitrary ">2.5 m argmax" rule.

**Posture compensation = PCA.** The chest-displacement I/Q arc is rotated onto its principal axis; this is
exactly `pca1`, and it recovers breathing per radar (e.g. +0.08 CS over the naive axis).

**Dual-radar (COM/TV, 90°).** The two radars view the chest in quadrature (±90°), so at least one has good
radial projection regardless of facing. All four candidates (COM/TV × LOS/Ghost) are extracted and
visualized over full time.

**Breathing-rate (BPM) recovery works, and the near-field system is validated.** On genuinely still
moments (body speed < 0.03 m/s) the near-body signal recovers rate within ±3 bpm for ~75% of windows
(≈6× chance). Across subjects the dual-radar oracle gives **median BPM error ~20%**, and the best
subjects (e.g. sub35, sub10) reach **±15% for 65–68%** of windows.

## 2. Evaluation methodology (rigorous)

- **Lag-aligned CS** (|Pearson|, ±2 s) as the waveform-similarity metric.
- **Null control**: CS against a +20 s-shifted same-subject GT (same rate/morphology, wrong phase).
  `real-gain = CS − null`. Also a permutation-null (K random shifts → p-value).
- **BPM RRAE** = |BPM_pred − BPM_GT| / BPM_GT, and `within-15%` = fraction of windows within 15%.
- Subject-wise LOSO for any learned component.

This two-metric discipline (waveform CS **with null** + rate BPM) is a key contribution: it separates
genuine recovery from breathing-band coincidence.

## 3. Key result — rate is recoverable, waveform is the hard part

The most important finding: **CS and BPM decouple.**
- Waveform CS sits near a **null floor of ~0.5** on walking data (any two breathing-band signals correlate
  ~0.5), so waveform `real-gain ≈ 0` — confirmed on our data, the processed data, and even on the
  RF-Carer authors' own walking recordings.
- **Rate (BPM) is a separate, recoverable quantity** — regular breathers (sub35) reach ±15% for 68% of
  windows. A subject can have mediocre waveform CS yet excellent rate.

Implication: the realistic product is a **confidence-aware respiration-rate monitor**, not pixel-level
waveform reconstruction. Note also that high inter-radar agreement can lock onto gait rather than breath,
so agreement alone is not a usable confidence signal.

## 4. Limitations → Future Work

1. **Continuous-walking protocol is the bottleneck (not the method).** The near-field system works when
   still, but only ~0.5% of windows are still enough (< 0.03 m/s) in these continuous-walking recordings.
   *Future:* protocols with more quasi-stationary segments, or explicit motion-gated / abstaining output.

2. **Waveform reconstruction is information-limited in the walking regime.** No hand rule (per-window /
   per-subject selection, dual-radar fusion, motion gating) beats always-Ghost, and learned U-Net
   regression collapses (mode-averaging). *Future:* a **conditioned Diffusion Transformer** that
   (a) takes all four candidates + I/Q + covariates, (b) models the conditional distribution instead of
   collapsing, and (c) exposes **sample-variance as a calibrated confidence** for gating. Evaluation must
   use BPM + null (not CS) and a conditional-vs-unconditional ablation to rule out prior hallucination.

3. **Sync uncertainty (~seconds).** GT↔UWB alignment is only known to a few seconds and cannot be pinned
   by cross-correlation alone (breathing periodicity). *Future:* a hardware sync marker per recording.

4. **Nearable vs wearable gap.** A near-field radar cannot match a chest-belt (BIOPAC) exactly; the goal is
   context/confidence-aware monitoring, acknowledging the Tx/Rx and geometry constraints.
