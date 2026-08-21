# Warmup and loss-curve audit — act_delta / act_quality / act_dit / patch_policy / vita

Date: 2026-08-21
Scope: LR schedule (warmup) per policy, and whether the fast training-loss drop indicates a problem.
Sources: `lerobot/src/lerobot/policies/*/configuration_*.py`, `lerobot/src/lerobot/optim/{factory,schedulers,optimizers}.py`,
`/mnt/robot_platform/jobs/*/log.jsonl` + `job.json`.

---

## 1. Answer: which policies use warmup

The LR schedule comes from `config.get_scheduler_preset()`, built in `optim/factory.py:41`
(`cfg.scheduler.build(...) if cfg.scheduler is not None else None`). Returning `None` means
**no scheduler at all** — constant LR from step 0, no warmup, no decay.

| policy | `get_scheduler_preset()` | warmup | LR schedule | peak LR |
|---|---|---|---|---|
| `act_delta` | `None` (`configuration_act_delta.py:164`) | **no** | constant | 1e-5 |
| `act_quality` | inherited from `ACTConfig` → `None` (`act/configuration_act.py:159`) | **no** | constant | 1e-5 |
| `act_dit` | inherited from `ACTConfig` → `None` | **no** | constant | 1e-4 |
| `patch_policy` | `None`, explicitly (`configuration_patch_policy.py:360`) | **no** | constant | 5.5e-5 |
| `vita` | `DiffuserSchedulerConfig(name="cosine", num_warmup_steps=500)` (`configuration_vita.py:232`) | **yes, 500 steps** | linear warmup → cosine decay to ~0 | 1e-4 |

**Only `vita` has warmup.** The other four run a flat LR for the whole job.

Confirmed against the logs (all runs used `use_policy_training_preset: true`, `optimizer_lr: null`,
so the preset above is what actually ran):

- `vita_..._2026-08-17_23-19-27`: `lr:7.5e-05` at step 1K, `1.0e-04` from step 2K, decaying to `9.2e-10` at 300K.
  (7.5e-5 is the *window average* over steps 0–1000: 500 warmup steps at mean 0.5× plus 500 at 1.0× = 0.75× — the
  500-step warmup is doing exactly what the config says.)
- `act_dit_...`: `lr:1.0e-04` at every one of 300 logged points.
- `patch_policy_...`: `lr:5.5e-05` flat.
- `act_quality_...` and `act_delta_...`: `lr:1.0e-05` flat.

Note `act_dit` does have an internal **weight EMA** with its own warmup ramp
(`modeling_act_dit.py:100`, `decay = min(0.9999, (1+n)/(10+n))`), and the 2026-08-20 runs had
`use_ema: true`. That is a weight averager, not an LR schedule — it does not affect the training loss curve
(the loss is computed on the live weights), only what is sampled at eval/deploy.

---

## 2. Is the fast loss drop a problem?

**No — not by itself, and it is not new to these policies.** The upstream `act` baseline on the same
dataset has the identical curve shape.

Fraction of the *entire* run's loss drop already completed by the end of epoch 1:

| run | loss @ep0.05 | @ep0.5 | @ep1 | @ep4 | final | % of total drop by ep1 |
|---|---|---|---|---|---|---|
| `act` (baseline, 200K) | 1.801 | 0.183 | 0.129 | 0.066 | 0.042 @ep10.6 | **99%** |
| `act_delta` (200K) | 3.077 | 0.331 | 0.198 | 0.100 | 0.046 @ep21.8 | **95%** |
| `act_quality` (216K) | 1.251 | 0.169 | 0.121 | 0.062 | 0.042 @ep9.4 | **97%** |
| `act_dit` (300K, flow-matching) | 0.373 | 0.077 | 0.064 | 0.046 | 0.025 @ep16.0 | **89%** |
| `patch_policy` (128K, diffusion head) | 0.184 | 0.054 | 0.043 | 0.024 | 0.018 @ep6.8 | **85%** |
| `vita` (300K) | 0.354 | 0.047 | 0.039 | 0.023 | 0.010 @ep16.0 | **92%** |

The baseline drops *fastest* of all six (99%). So whatever is causing the steep early descent is a
property of this dataset + the ACT-family loss, not something the new policies introduced. Three
concrete reasons it is expected here:

1. **The starting loss is a scale artifact, not information.** All these losses are on
   *normalized* actions. At init the network outputs ~0 while targets are unit-variance, so the L1
   losses (`act`, `act_delta`, `act_quality`) start near the mean absolute value of the data,
   inflated further by `kl_weight: 10.0` on an untrained CVAE latent. Falling from 3.0 to 0.2 is
   mostly the model learning the *mean action* and the KL term collapsing — a few hundred steps of
   work, not evidence of generalization. The MSE/velocity losses (`act_dit`, `patch_policy`, `vita`)
   start an order of magnitude lower precisely because they never had that headroom.

2. **`grad_clip_norm` defaults to 10.0** (`optim/optimizers.py:91`) and is *not* overridden by any of
   these configs. The ACT-family runs log `grdn: 65` at step 1K, decaying past 10 only around
   step ~5–8K. The first few thousand steps are therefore clipped — which is already acting as an
   implicit step-size limiter, and is why the absence of warmup has not blown any of these runs up.

3. **The teleop action target is close to the observed state.** Per this repo's own conversion
   semantics, `action` is the first `joint_cmd` after the observation tick, and under
   `joint-state-fill` / `--missing-topic-policy fill` some action columns are literally an identity
   copy of `observation.state`. A network with `observation.state` on its input can reach a low L1
   by learning "output roughly the current pose" — which is exactly a loss that falls hard in the
   first epoch and then flattens.

### What *is* a real problem

**None of these runs measures generalization at all.** Every job.json in scope shows
`env_eval_freq: 0`, and every log shows `'eval_steps': 0`. `configs/train.py:108` supports a
held-out eval loss (`eval_steps > 0` + `dataset.eval_split > 0`), but LeLab's job submission never
sets either field (`grep -rn "eval_steps\|eval_split" apps/lelab` → no hits). So the only number on
these curves is training loss, and a training loss that flattens by epoch 1 and then creeps down for
another 10–20 epochs is **indistinguishable from memorization**. The three ACT-family runs at
epoch 10–22 on a few hundred episodes are in exactly the regime where that matters.

Secondary observations:

- `act_dit` at LR 1e-4 with **no warmup and no decay** is the one configuration where the missing
  warmup is a genuine (if so-far-benign) risk: it is 10× the ACT preset, on a noise/velocity target,
  and unlike `vita` it never anneals. Its final loss is still improving at step 300K with a flat LR,
  which is the signature of a run that would have converged lower with a cosine tail.
- The `act_quality` job at `2026-08-20_08-54-41` is marked `state: failed / exit_code 1`, but
  `slurm.out` shows `JOB 117 ... CANCELLED ... DUE to SIGNAL Terminated` at step 216K/250K — an
  external cancellation, not a training error. Its 21 checkpoints are valid.

---

## 3. Recommendations, in priority order

1. **Add a held-out eval split.** Plumb `dataset.eval_split` / `eval_steps` through the LeLab job
   config (they already exist in `configs/train.py`, they are simply not exposed). Without this,
   no statement about any of these five policies vs. the baseline can be supported by the loss curve.
   This is the only item that changes what you can conclude.
2. **Give `act_dit` a scheduler.** It is the only one running a 10× LR flat to the end.
   `CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=500)` is already registered in
   `optim/schedulers.py:104` — a five-line `get_scheduler_preset()` in
   `configuration_act_dit.py`, matching what `vita` already does.
3. **Leave the other three alone.** `act_delta` / `act_quality` at 1e-5 with clipping at 10 are the
   upstream ACT recipe unchanged; adding warmup there changes a baseline you are comparing against
   for no measured benefit. `patch_policy`'s constant LR is a deliberate match to its reference
   implementation and is documented as such at `configuration_patch_policy.py:361`.
4. **Sanity-check the identity-copy shortcut** on the datasets in use: `meta/conversion_manifest.json`
   → `audit.hold.joint_state_fill_rows` and `audit.end_effector_action_source`. If a large fraction
   of rows have action columns copied from the observation, the low training loss is partly free and
   the fast drop is fully explained.
