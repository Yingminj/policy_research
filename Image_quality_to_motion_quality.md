# From Image Quality to Action Quality: Literature Review and Experimental Design

**Scope.** What is known about how lossy image/video compression affects learned
robot manipulation policies, how that relates to the pixel- and feature-level
results already produced in `robot_data_platform/test_raw_jpeg`, and what
experiment would actually close the remaining question.

Compiled 2026-08-06. Sources verified by direct fetch except where marked
*(unverified)*.

---

## 1. What we already have

`test_raw_jpeg/REPORT.md` measures a two-stage compression cascade on one
36.8 s / 1103-frame recording of the `express` task, using the single
uncompressed `/quad_tile` bag as a pixel-level ground truth. Three sources
(raw, JPEG q100, JPEG q80) × three storage levels (h264 CRF 0/20/30), with
byte-identical joint state and action arrays across all nine legs — only the
image bytes differ.

Established there:

| finding | number |
|---|---|
| q80 vs raw, mosaic | 42.296 dB PSNR, 0.9747 SSIM |
| q80 loss concentrates on wrists | head 45.95 dB vs wrist_L 38.99 dB (6–7 dB gap, mosaic stores head 2× upscaled) |
| q80's marginal cost at CRF 20 | −1.181 dB vs raw source |
| q80's marginal cost at CRF 30 | −0.359 dB |
| q80's marginal cost at CRF 0 | −4.785 dB (and *larger* files) |
| DINO CLS cosine, jpeg80_crf20 | 0.938–0.976 across three backbones |
| PSNR ↔ CLS cosine | Spearman ρ = 0.927, Pearson r = 0.753 |
| GOP=2 I/P oscillation at CRF 20 | 1.3–1.4 dB, alternating every frame |

The report's own stated limitation is the open question: *pixel fidelity and
policy success rate are not the same thing.* Everything below is about closing
that gap.

---

## 2. Nothing in this cluster has tested it

Audit of `/mnt/robot_platform/jobs/` (7 jobs) and the 6 datasets they consume:

| dataset | eps | codec | CRF | trained |
|---|---:|---|---:|---|
| express | 51 | h264 | 0 | 100k steps, ACT |
| pack_airpods_lerobot | 40 | h264 | 20 | 300k steps, running |
| tea_2_lerobot | 53 | h264 | 20 | — |
| dex_stack_box_mcap_le | 100 | h264 | 30 | — |
| jjj_le | 50 | h264 | 30 | — |
| peg_optical_module_0721_120 | 120 | av1 | — | 1k steps |

The CRF axis is spanned (0 / 20 / 30 / AV1) but each level sits on a different
task, robot configuration and episode count, so compression is fully confounded
in every existing run. Every job used `seed 1000`; the seed has never been
varied, so the noise floor that any compression effect must clear is unmeasured.

`dataset/sc3_lerobot` (h264 CRF 0) and `dataset/sc3_lerobot_crf30` (av1 CRF 30)
are the same 186 episodes / 167,709 frames at two quality levels — a genuine
matched pair, but never trained on, and the two differ in codec *and* CRF
simultaneously.

**Usable asset:** `act_express_2026-08-04_21-08-58-027924` is ACT trained 100k
steps on `express` (51 episodes, q80 source, h264 CRF 0) — precisely the
`jpeg80_crf0` cell of the test grid. Its config has `chunk_size=100`,
`n_action_steps=100`, `temporal_ensemble_coeff=None`, `vision_backbone=resnet18`.

---

## 3. Literature

### 3.1 The reference benchmark stops at pixels

LeRobot's own video benchmark ([PR #282](https://github.com/huggingface/lerobot/pull/282))
sweeps `libx264`/`libx265`/`libsvtav1` × CRF {0, 5, 10, 15, 20, 25, 30, 40, 50, None}
× GOP {1…40, None} × {yuv444p, yuv420p}, and reports `avg_mse`, `avg_psnr`,
`avg_ssim` plus encode/decode timing. **No downstream model metric.**

This matters for positioning: `test_raw_jpeg` is already at parity with the
reference benchmark on the pixel axis and past it on the feature axis. It also
corrects a citation error in the field — [Robo-DM](https://arxiv.org/html/2505.15558v1)'s
related-work section states that "LeRobot empirically evaluates how lossy video
compression parameters in FFmpeg affect robot policy accuracy," which overstates
what that benchmark computes.

### 3.2 The one paper that goes downstream, and its limits

**Robo-DM** (arXiv [2505.15558](https://arxiv.org/html/2505.15558v1)) is the
closest published work. Two relevant experiments:

- *Octo fine-tuning:* validation MSE 1.86 (lossless) → 1.91 (lossy), reported as
  a 2.6% increase. One condition pair, no seed replication.
- *Physical Franka pick-and-place:* 335 trajectories at 75.3× lossy compression,
  **15/15 success**.

The hardware result cannot support the conclusion it is used for. At n=15 with
15 successes, the 95% Clopper–Pearson lower bound is ≈78% — the experiment
cannot distinguish "no effect" from a drop to roughly 80%. It rules out
catastrophic degradation and nothing finer. There is no train/deploy mismatch
condition and no recording-side compression stage.

Robo-DM also reports up to 70× size reduction versus the format Open X-Embodiment
uses for distribution, which is the practical motivation for the whole question.

### 3.3 Compression degrades vision tasks — established outside robotics

**"A Perspective on Deep Vision Performance with Standard Image and Video Codecs"**
(arXiv [2404.12330](https://arxiv.org/abs/2404.12330)) examines JPEG and H.264
across classification, localization and dense prediction. From the abstract:

> "We find that using JPEG and H.264 coding significantly deteriorates the
> accuracy across a broad range of vision tasks and models. For instance, strong
> compression rates reduce semantic segmentation accuracy by more than 80% in
> mIoU."

The key structural finding is that **dense prediction degrades far faster than
classification**. This is the most transferable result for manipulation: a
visuomotor policy reading wrist cameras for contact-level geometry is closer to
dense prediction than to classification, which argues the classification-derived
intuition ("q80 is basically free") is the wrong prior.

*(unverified)* Search snippets attribute to this paper a 3.1% ResNet-50 top-1
drop at JPEG q80 and 1.9% at q90; the PDF did not extract cleanly and these
numbers were not confirmed in-text. Verify before citing.

Counter-literature worth noting for balance: several works report compression
*helping* classification ([Compression Helps Deep Learning in Image
Classification](https://www.mdpi.com/1099-4300/23/7/881)), typically via a
denoising/regularization effect. The effect is task- and dataset-dependent and
does not generalize to fine-grained spatial tasks.

**ImageNet-C** includes JPEG compression as one of its 15 standard corruption
types, so compression-as-corruption is long-established methodology.
**RobustNav** ([arXiv 2106.04531](https://arxiv.org/pdf/2106.04531)) ports this
to embodied navigation with 7 visual corruptions. On the manipulation side,
LIBERO-Plus and VLATest cover layout, appearance, viewpoint, lighting and sensor
noise — but codec artifacts specifically are not in these suites.

### 3.4 Pixel metrics don't predict machine task performance — a whole field says so

**Video Coding for Machines (VCM)** exists as an MPEG standardization line
precisely because PSNR/SSIM/VMAF do not align with machine task performance.
Task metrics (mAP@[0.5:0.95], mIoU, MOTA) replace pixel fidelity in the
evaluation, and codec designs that optimize rate–task tradeoffs differ
structurally from human-oriented codecs
([survey](https://arxiv.org/pdf/2001.03569)).

VCM also anticipates a second problem the `test_raw_jpeg` report ran into
independently: single-model evaluation is biased. **Satisfied Machine Ratio**
(arXiv [2211.06797](https://arxiv.org/html/2211.06797)) aggregates quality
judgments across many machine models for exactly this reason. The report's
observation that CLS cosine at `raw_crf30` ranges from 0.8628 to 0.9169 across
three backbones — against a single PSNR value — is the same phenomenon, and the
three-backbone design is the right response.

**Implication for the report:** its §阶段三 conclusion (PSNR orders correctly but
cannot serve as an acceptance threshold — ρ=0.927 vs r=0.753, with 30–100× more
CLS movement per dB below 40 dB than above 43 dB) is an independent rediscovery
of the VCM premise on robotics data. That is a citable framing, not a caveat.

### 3.5 Offline action error is a weak proxy for closed-loop success

This is the sharpest constraint on any cheap experiment.

**RoboMimic** ([study](https://robomimic.github.io/study/)) found the
best-validation-loss checkpoint is **50–100% worse** than the best-performing
checkpoint. Validation loss is a surrogate; policies are not selected by it in
practice for good reason.

**Critical Interval MSE** (arXiv [2606.29898](https://arxiv.org/pdf/2606.29898))
diagnoses why: plain action MSE weights every timestep equally, while errors
during task-critical phases (approach, contact, grasp, insertion) dominate
outcomes. CI-MSE weights errors by criticality and reports substantially higher
correlation with closed-loop success across ManiSkill2, LIBERO and RoboArena.
*(The specific correlation coefficients did not extract from the PDF — fetch the
tables before citing numbers.)*

**Consequences for our design:**

1. Validation action MSE is usable as a *screen* — it can establish that two
   compression conditions are indistinguishable — but cannot certify one as
   better.
2. Any Δaction metric should be weighted toward the grasp and insertion windows
   of the `express` task rather than averaged over all 1103 frames. A 2 mm
   commanded-TCP shift during free transport is irrelevant; the same shift at
   contact is not.

### 3.6 Closed-loop evaluation statistics

Current practice is badly underpowered, and this is now well documented.

- Typical real-robot comparisons use 20–30 trials; most fall in a 10–50 rollout
  regime ([N-SCORE](https://arxiv.org/html/2603.13616)).
- Precision cost: an observed 90% success rate at 70 rollouts gives a 95%
  Clopper–Pearson interval of [80.5%, 95.9%] — 15.4 pp wide. Tightening to
  ±2 pp requires ~1,030 rollouts, roughly 15× more.
- Fixed-sample two-proportion tests at α=0.05, 80% power: detecting 90%→70%
  needs ~62 trials per arm; 90%→80% needs ~200 per arm.

Two methods reduce this materially and both apply directly:

**STEP** (arXiv [2503.10966](https://arxiv.org/html/2503.10966)) — sequential
testing with control-theoretic decision regions over binary outcomes; stops
early once evidence is sufficient. Approaches oracle SPRT performance for 5–10 pp
gaps, where asymptotic methods fail in finite samples.

**N-SCORE** (arXiv [2603.13616](https://arxiv.org/html/2603.13616)) — sequential
comparison over *graded* outcomes. Its framing of why binary scoring wastes data:

> "a policy that completes 90% of the task is clearly better than a policy that
> is frozen the whole time, yet their success rates would be identically 0%."

Reported savings: up to 70% fewer evaluations versus batch methods using partial
credit, up to 50% versus binary sequential procedures; on hardware, 50 nominal
rollouts reduce to ~38–42; on RoboArena real-world data, 1,419 trials versus
1,881 to separate four policies.

**Consequence:** a graded rubric (approach / grasp / transport / place, weighted)
plus sequential stopping is the difference between a feasible closed-loop
experiment and an infeasible one. Binary success at n=15, as in Robo-DM, is not
an experiment.

### 3.7 Train/deploy codec mismatch — named failure mode in adjacent fields

Not yet studied in robotics, but well characterized elsewhere. In synthetic-image
detection and document forensics, the mismatch between the narrow set of JPEG
quantization tables seen in training and the heterogeneous compression profiles
of real deployment pipelines is a *primary* source of generalization gap
([DocQT, arXiv 2605.19688](https://arxiv.org/html/2605.19688)). The standard
mitigation is JPEG-compression augmentation during training.

This maps directly onto our deployment: the policy trains on h264-decoded frames
and infers on live q80 JPEG frames from `realsense_node.cpp`. Same mechanism,
untested in this domain.

Adjacent but distinct: **Quantization-Aware Imitation Learning**
(arXiv [2412.01034](https://arxiv.org/pdf/2412.01034)) compresses the *model*,
not the data.

---

## 4. Gap analysis

Published:

- pixel-level rate–distortion for robot dataset codecs (LeRobot benchmark)
- one downstream data point at one compression level (Robo-DM), underpowered
- compression degrades vision tasks, dense prediction worst (2404.12330)
- pixel metrics ≠ machine task metrics (VCM, an entire subfield)
- offline action error ≠ closed-loop success (RoboMimic, CI-MSE)
- how to run adequately powered policy comparisons (STEP, N-SCORE)

Not found in the literature:

1. **The recording-side × storage-side cascade.** Every study treats compression
   as one stage. Production pipelines have two — camera-side JPEG then
   dataset-side video codec — and the report's central finding is that they
   interact non-monotonically (q80→CRF 0 is strictly dominated: larger files
   than raw→CRF 20 at equal quality, because low CRF faithfully preserves JPEG
   block noise).
2. **Train/deploy codec mismatch for robot policies.** Named and studied in
   forensics, absent in robot learning, and structurally present in every
   deployed LeRobot system.
3. **Per-camera asymmetry under a mosaic layout.** The head/wrist 6–7 dB split
   caused by 2× upscaled head tiles, with the wrists — the cameras that matter
   for fine manipulation — taking the loss. Follows from the recording format,
   not from the codec.
4. **Systematic per-frame quality alternation from GOP=2.** A 1.3–1.4 dB I/P
   oscillation at 15 Hz baked into every LeRobot dataset stored at CRF 20. Its
   effect on temporal modeling is unexamined anywhere.

Items 1–2 are the publishable core. Item 4 is a LeRobot-wide observation.

---

## 5. Resulting experimental design

Three rungs, cheapest first. Rung 1 uses only assets already on disk.

### Rung 1 — action sensitivity of a fixed policy (~1 GPU-hour)

Run the frozen `act_express` checkpoint over all 11 image variants × 1103
frames, holding state input identical. ACT zeroes the VAE latent at inference,
so it is deterministic: every output difference is attributable to pixels, with
no seed noise.

Measure:

1. **Δaction in radians**, per joint, mean / p95 / max. Two reference points:
   vs `jpeg80_crf0` answers "what breaks if I change storage under a deployed
   policy"; vs `raw` answers "what did compression cost".
2. **Normalized by the policy's own error** `E = ‖predicted − ground-truth teleop‖`
   on the reference variant. The decision quantity is Δ/E. Without this
   denominator a radian figure is uninterpretable.
3. **Criticality-weighted**, per §3.5 — grasp and insertion windows separated
   from free transport.
4. **By chunk position**, k=0..9 versus the tail. This configuration has
   `n_action_steps=100` and `temporal_ensemble_coeff=None`: one observation
   determines 3.3 s of motion at 30 fps, with no temporal averaging to attenuate
   per-frame compression noise. This policy is unusually exposed, and that is
   worth reporting independently.
5. **Temporal structure of Δ** — specifically whether the GOP=2 15 Hz
   oscillation propagates into the action. i.i.d. frame noise is benign;
   correlated noise is not.
6. **Gripper channel separately** — a flip differs qualitatively from a joint
   wobble.
7. **Task-space conversion** via FK to mm of commanded TCP displacement, if the
   URDF is available.

Cheap addition: repeat the DINO cosine analysis on the policy's own ResNet18
backbone, testing whether generic-backbone drift is a valid proxy for the
policies actually being trained — currently an untested assumption in §阶段三.

Rung 1 measures the *mismatch* axis only. A policy trained on CRF-30 data may
learn compression-robust features while a CRF-0-trained policy fed CRF-30 images
fails. That requires training.

### Rung 2 — train the grid; open-loop error and the mismatch matrix

Measured cost from the existing job: 100k steps, batch 8, 49,843 frames →
**3h08m, 5.8 GB VRAM**. Two GPUs available.

| variant | data required | runs | GPU-h | wall (2 GPU) |
|---|---|---:|---:|---:|
| **2a** CRF axis: q80 × CRF {0,20,30,40} × 3 seeds | re-convert 51 existing bags | 12 | 38 | ~19 h |
| **2b** full grid: {raw, q100, q80} × CRF {0,20,30} × 3 seeds | ~50 raw episodes ≈ 310 GB | 27 | 85 | ~43 h |

2a answers "how cheap can storage get" but cannot answer "what did q80 cost me" —
only one raw episode exists, and one episode does not train a policy. 2b requires
a raw recording campaign (6.2 GiB per 37 s episode); the payoff is that
`02_build_jpeg_bag.py` then synthesizes q100/q80 from those exact frames, giving
identical trajectories across conditions — an internal validity almost no
published compression study achieves.

Non-negotiable design points:

- **≥3 seeds per condition.** The seed spread is the noise floor; differences
  below it are unmeasurable at this budget, and for CRF 20 vs CRF 0 they
  plausibly will be. No existing job here has ever varied the seed.
- **Pre-registered non-inferiority margin**, declared before running.
- **Mismatch matrix** `M[i][j]` = policy trained on condition *i*, evaluated on
  *j*. Off-diagonal minus diagonal is the domain gap. Column `j = jpeg_q80` is
  the production column — that is what the camera delivers at deploy time.
- Per §3.5, val MSE screens but does not certify. Report CI-MSE-style weighted
  error alongside it.

### Rung 3 — closed-loop success

The only metric that answers the question, and the reason rungs 1–2 exist is to
arrive here with two candidates rather than nine.

Protocol, following §3.6:

- Graded rubric with partial credit (approach / grasp / transport / place),
  not binary success — this is where the 50–70% sample savings come from.
- Sequential stopping (STEP for binary, N-SCORE for graded) rather than a fixed
  batch.
- Fixed initial-state jig; interleaved randomized condition order; operator blind
  to condition.
- Intervention count and time-to-completion as continuous secondaries.

Budget expectation: ~40 s per trial plus reset; ~40 graded trials per condition
under sequential stopping, versus ~62 binary trials fixed-sample for a 20 pp
effect. Anything smaller than ~10 pp is out of reach on hardware and should be
settled at rungs 1–2 or in simulation.

If compression studies recur, a simulator for the `express` task is the highest-
leverage infrastructure investment available — it converts rung 3 from hours of
robot time into minutes of GPU time, and the evaluation-in-simulation literature
([arXiv 2510.04354](https://arxiv.org/html/2510.04354)) addresses the fidelity
caveat directly.

---

## 6. Positioning for publication

The pixel and feature work is done and is already past the field's reference
benchmark. What would make it a contribution rather than an internal report:

1. Rung 1 — the train/deploy mismatch measurement. ~1 GPU-hour, data on disk.
2. Rung 2a — the CRF axis at training scale with seed replication. ~19 h wall,
   no new recording.
3. Rung 2b + rung 3 — the raw→q80 question and closed-loop confirmation. Requires
   a recording campaign and robot time.

(1) and (2) together already constitute the first controlled study of
compression-induced train/deploy mismatch in robot learning, with an internal
validity — byte-identical actions across conditions — that the existing
literature does not have.

Note that `test_raw_jpeg/{intermediate,results,lerobot,bags}/` have been deleted
and `run_all.sh` referenced in the README does not exist; stages 1–3 must be
re-run before rung 1 has frames.

---

## Sources

**Directly relevant, verified**

- [Improve video benchmark · PR #282 · huggingface/lerobot](https://github.com/huggingface/lerobot/pull/282) — reference codec benchmark, pixel metrics only
- [Robo-DM: Data Management For Large Robot Datasets](https://arxiv.org/html/2505.15558v1) — the only downstream compression evaluation; underpowered
- [What Matters in Learning from Offline Human Demonstrations (robomimic)](https://robomimic.github.io/study/) — validation loss a poor predictor
- [Critical Interval MSE](https://arxiv.org/pdf/2606.29898) — criticality-weighted offline validation
- [Is Your Imitation Learning Policy Better than Mine? (STEP)](https://arxiv.org/html/2503.10966) — sequential policy comparison
- [Beyond Binary Success (N-SCORE)](https://arxiv.org/html/2603.13616) — graded outcomes, sample efficiency
- [A Perspective on Deep Vision Performance with Standard Image and Video Codecs](https://arxiv.org/abs/2404.12330) — JPEG/H.264 across vision tasks

**Context**

- [Video Coding for Machines: A Paradigm of Collaborative Compression and Intelligent Analytics](https://arxiv.org/pdf/2001.03569)
- [Perceptual Video Coding for Machines via Satisfied Machine Ratio Modeling](https://arxiv.org/html/2211.06797)
- [RobustNav: Benchmarking Robustness in Embodied Navigation](https://arxiv.org/pdf/2106.04531)
- [Open X-Embodiment](https://arxiv.org/html/2310.08864v4)
- [Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators](https://arxiv.org/html/2510.04354)
- [DocQT: JPEG Quantization Table Mismatch](https://arxiv.org/html/2605.19688) — train/deploy codec gap in an adjacent field
- [Compression Helps Deep Learning in Image Classification](https://www.mdpi.com/1099-4300/23/7/881) — counter-evidence
- [Quantization-Aware Imitation-Learning](https://arxiv.org/pdf/2412.01034) — model compression, adjacent

**Searched, nothing found**

No study located on: codec artifacts in manipulation robustness suites; JPEG-quality
ablations for ACT or diffusion policies; recording-side × storage-side compression
cascades; or compression train/deploy mismatch matrices for robot policies. Absence
of search hits is not proof of absence — a proper database sweep (IEEE Xplore, ACM DL,
Semantic Scholar API) is warranted before asserting novelty in a submission.
