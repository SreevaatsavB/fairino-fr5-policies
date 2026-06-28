# 8. The Fix — what we actually changed (branch `fix/dino-act-cvae-probe`)

Files 1–7 *diagnose* the DINOv2 ACT failure. This file documents **what we did about it** on the
`fix/dino-act-cvae-probe` branch: every change, why, and how to verify it. It targets the two
*fixable in code* root causes — the **CVAE posterior collapse** and the **lack of a vision-vs-state
model-selection metric** — and ports the real DINOv2 model so the fixes land on the actual failing
setup. Scope is **train + offline probe** (no deploy changes); the grasp gate is documented only
(file 7), and gripper-dropout was intentionally dropped as over-engineering.

> One-line: keep the CVAE latent **alive** so ACT stops mode-averaging, and add a probe that
> measures whether the policy uses the **cameras** vs its own **joint state** so you can pick models
> by that instead of by validation loss.

---

## 8.1 The CVAE KL fix (the headline change)

**Problem (file 3 §3.2 / file 6):** with `kl_weight=10` baked into lerobot's loss, the KL term
collapsed to ~0 in every run (`0.21 → 0.0001`). A dead latent = ACT degenerates into a plain
mean-regressor = the object-blind "average" grasp.

**What we changed:** `policies/act/model.py` `ACT.forward` no longer uses lerobot's pre-combined
loss. It replicates lerobot's forward (the `OBS_IMAGES` stack + `self.model(batch)` call) but
**recomputes the KL term** with three anti-collapse mechanisms. Plus new config keys.

```python
# per-latent-dim KL (shape (B, 32))
kld_dim = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
# (2) FREE-BITS: floor each dim at lambda before summing
kld = kld_dim.clamp(min=self.kl_free_bits).sum(-1).mean()
# (1) ANNEALING: ramp the weight 0 -> kl_weight over kl_warmup_steps
kl_w = self.kl_weight * min(1.0, _kl_step / self.kl_warmup_steps)
loss = l1 + kl_w * kld          # (3) lower kl_weight (10 -> 0.5)
```

Config (`config.yaml` + the dino configs): `kl_weight: 0.5`, `kl_warmup_steps: 2000`,
`kl_free_bits: 0.03` (smoke/local use smaller warmups).

> **Honesty note (read this).** The CVAE change is the **most-deviating, least-certain** part of
> this branch — the ACT paper uses `kl_weight = 10` and *deliberately* keeps the latent **weak**
> (because `z = 0` at inference). The well-evidenced root cause is the **proprioceptive shortcut**
> (the measured 2.2× state>vision), not the CVAE. So the free-bits floor was deliberately set
> **gentle** (`0.03/dim ≈ 1 nat`, within the healthy 0.5–5 KL range) — just enough to stop KL hitting
> exactly zero, *not* enough to force a strong latent that would mismatch `z = 0` at deploy. Treat
> the KL change as a small, **evidence-gated** tweak: watch the training KL curve and the probe,
> and if it doesn't help, set `kl_weight: 10` + remove free-bits to fall back to the proven recipe.

### Why three mechanisms — and what each one is

The KL term measures how much information the latent `z` carries. Reconstruction wants `z` to carry
**lots** (helps predict the action); the KL wants it to carry **none** (stay at the prior). At
`kl_weight=10` the KL side won outright → collapse. The three fixes each weaken that pull:

**(1) KL annealing — "ease into the regularization."**
Don't apply the KL penalty from step 1; start it at **0** and ramp to the target over
`kl_warmup_steps` (2000). *Why:* at the start, `z` is useless noise (the decoder hasn't learned to
use it), so full KL pressure makes the optimizer take the cheap path — zero `z` out — and then the
decoder never learns to use it (chicken-and-egg → permanent collapse). With the weight ≈ 0 early,
the model is **free to learn to use `z`** first (reconstruction makes it informative); *then* the
KL is gently turned on so `z` stays well-behaved but alive.
- At step 0: `kl_w = 0` → loss = pure reconstruction (verified: `loss == l1` at step 0).
- At step ≥ 2000: `kl_w = 0.5` (full target).
- `_kl_step` is a saved buffer that advances on each training step (resume-safe).
- *Analogy:* teach someone to **use** a tool before grading them on a tidy workbench.

**(2) Free-bits — "a tax-free allowance per latent dimension."**
Give each of the 32 latent dims a small amount of KL it may use **for free** (no penalty). Floor the
per-dim KL at `λ = 0.1`. Below the floor the optimizer gets **no reward** for shrinking further, so
it can't squeeze a dim to zero — each dim keeps ≥ λ nats of information.
- `kld_dim.clamp(min=0.1)` replaces any sub-0.1 dim with a flat `0.1`; a flat value has **zero
  gradient**, so there is no force pushing it lower. The penalty only "bites" on info **above** 0.1.
- *Analogy:* a tax-free allowance — each dim carries 0.1 nats tax-free; only info above that is taxed.

**(3) Lower weight.** `kl_weight: 10 → 0.5` — turn the overall KL pull down so the "use `z`" force
isn't overwhelmed. `10` was the single biggest collapse trigger.

### Why a non-zero KL fixes the object-blind grasp
A **live** latent means ACT can put "which demonstration mode this is" into `z` and, at inference
(`z = 0`), **commit to one mode** instead of averaging all of them. Averaging is exactly what
produced the phantom "mean" reach + mid-air grasp (file 2 §2.3). Keep `z` alive → ACT can pick a
mode → less object-blind averaging.

### How to verify (Linux/GPU box)
- Every run writes a per-epoch **`<checkpoint_dir>/metrics.csv`** with the columns
  `train_l1, train_kl, val_l1, val_kl, train_val_gap, grad_norm, kl_weight_eff, lr, seconds`.
  Plot it: **`python tools/plot_metrics.py <checkpoint_dir>`** — it saves a PNG and prints a
  verdict. The KL should **level off above zero** (~0.5–5), not crash `0.21 → 0.0001`. (The
  plotter flags `final KL >= 0.1 -> HEALTHY` vs `-> COLLAPSED`.)
- Feed the **same** image with two different `z` samples → the predicted action chunk should
  **visibly differ** (proof `z` is being used).

### Logging captured for error analysis (next runs)
Every run also drops a one-time **`<checkpoint_dir>/run_info.json`** provenance snapshot:
timestamp, **git commit/branch/dirty**, exact resolved config, dataset root + `action_space` +
train/val frame counts + `frame_stride` + `image_size`, device, torch/python/platform, and the
trainable-param count. This ties any later deploy failure back to *exactly* the code, config,
data split, and environment that produced the checkpoint. The extra `metrics.csv` columns are
the in-training diagnostics: **`val_kl`** (rising while `train_kl` stays low ⇒ the latent is
overfitting), **`grad_norm`** (a sudden spike precedes divergence), and **`train_val_gap`** (the
overfitting signal — file 9). Together with the existing checkpoint payload (`config`, `stats`,
`action_space`, `val_l1`) this is everything needed to error-analyse a run after the fact.

---

## 8.2 The DINOv2 2-camera ACT port

So the fixes apply to the *real* failing model (not the base ResNet ACT). Ported from vivek's `act`
branch, **without gripper-dropout** and merged carefully with this repo's delta-EEF work.

| File | Change |
|---|---|
| `policies/act/model.py` | `DinoV2Backbone` (frozen DINOv2 ViT-S/14, optional last-N trainable blocks) swapped in for lerobot's ResNet18; multi-camera `_make_batch` that accepts a single tensor **or** a `{camera_key: tensor}` dict; `predict_chunk()` for the probe. |
| `common/dataset.py` | per-camera image keys (`observation.images.<cam>`), `frame_stride`, and a `photometric` aug level. |
| `common/convert_episodes.py` | wrist+scene extraction + `--extract-stride` (kept the delta-EEF / `action_space` path). |
| `common/train.py` | passes a **single image tensor when there's 1 camera** (so the other 5 policies are untouched) and a **dict when there are 2** (ACT only); the DINOv2 backbone trains at **0.1× LR**. |
| `policies/act/config.joint_dino{,_ft,_prop}.yaml` | 2-camera DINOv2 configs (frozen / last-block-FT / proprio-dropout 0.5), all with the CVAE fix, no gripper-dropout. |

**Key design choice (lean & non-breaking):** the multi-camera convention only activates with 2+
cameras. The other policies always train on single-camera data, so they receive a single tensor
exactly as before — verified unchanged by `tools/test_inference.py` (all pass) and the diffusion
smoke test.

### Upgraded vision encoder — DINOv2 / DINOv3 / I-JEPA (configurable)

`DinoV2Backbone` was generalised to **`VisionBackbone`**, which loads any of three frozen
self-supervised ViTs and reshapes their patch tokens to lerobot's `(B, C, h, w)` feature map (it
drops any leading CLS/register tokens automatically, so the same code handles all three):

| `dino_backbone` | Loader | Size / patch | Notes |
|---|---|---|---|
| `dinov2_vits14` | `torch.hub` | 21M / p14 | the original (open, no auth) |
| **`dinov3_vits16`** *(new default for the dino configs)* | HF `AutoModel` | 21M / p16 | **upgrade** — stronger features, same footprint. **GATED**: `huggingface-cli login` with a token that accepted Meta's license. |
| `dinov3_vitb16` / `dinov3_vitl16` | HF `AutoModel` | 86M / 300M | bigger DINOv3 |
| `ijepa_vith14` | HF `AutoModel` | ~630M | open, but **GPU-only** (smallest public I-JEPA is ViT-H) |
| any `owner/model` | HF `AutoModel` | — | load any HF ViT id verbatim |

The internal net is kept as `self.dino` so `train.py`'s `backbone.dino` 0.1×-LR match still works.
The three `config.joint_dino*.yaml` now default to `dinov3_vits16`; set `dino_backbone:
dinov2_vits14` to revert (no auth). Verified by a mock test of the patch-extraction/reshape for all
three families (no weights downloaded locally); the actual DINOv3/I-JEPA load happens on the
Linux/GPU box (DINOv3 needs the HF login; I-JEPA needs the GPU).

---

## 8.3 The vision-vs-state probe + model selection

**Problem (file 3 §3.3):** checkpoints were picked by `val_l1`, which is *anti-correlated* with
deploy success (the deployed model had the **worst** val loss).

**What we added:** `experiments/shortcut_probe.py`. For each validation sample it takes the
predicted action chunk (`predict_chunk`, bypassing temporal ensembling) under input perturbations
and measures how much it MOVES:

- **zero-images** → `image_sensitivity` (does blanking the cameras change the action?)
- **shuffle-images** (state_A + another sample's images = the conflict-swap) → image *content* sensitivity
- **zero-state** → `state_sensitivity`
- **output_spread** (≈ 0 ⇒ a canned/constant trajectory) and `action_std` (the GT scale)

**Headline metric:** `state / image ratio = state_sensitivity / image_sensitivity`.
`>> 1` = object-blind (the deployed model was ~**2.2×**); `< 1` = image-dominant (good).

**Model selection:** `--checkpoints a.pt b.pt …` probes each, **ranks by vision-dominance**, and
recommends the most image-dominant — the metric to use **instead of `val_l1`**.

```bash
python experiments/shortcut_probe.py --ckpt policies/act/checkpoints_joint_dino/best.pt
python experiments/shortcut_probe.py --checkpoints run_a/best.pt run_b/best.pt run_c/best.pt
```

---

## 8.4 The grasp-success gate — documented, not coded

The mid-air-grasp fix needs a real grasp signal, but the FR5 gripper is command-only (no
width/force/position read). So it is **deferred** with a precise spec of the required signal and the
`deploy.py` insertion point — see [`07_grasp_gate_requirements.md`](07_grasp_gate_requirements.md).

---

## 8.5 Full change set (21 files, +2,106 lines)

```
code:    policies/act/model.py        DINOv2 + multi-cam + predict_chunk + CVAE fix
         common/dataset.py            per-camera keys + frame_stride + photometric
         common/convert_episodes.py   wrist+scene extraction + --extract-stride
         common/train.py              single-or-dict image passing + 0.1x DINOv2 LR
         common/smoke_test.py         new per-camera key assertions
         experiments/shortcut_probe.py  vision-vs-state probe + model selection
configs: policies/act/config.yaml / .local / .smoke   kl_weight 10->0.5 + warmup + free-bits
         policies/act/config.joint_dino{,_ft,_prop}.yaml   2-cam DINOv2 configs
docs:    docs/failure_analysis/01–08  the analysis + this fix log + grasp-gate spec
```

**Unchanged (no regressions):** the other 5 policies (diffusion, dit_flow, pi0/pi05/pi0_fast),
`deploy.py`, `common/proprio.py`.

---

## 8.6 What's verified vs what's left

**Verified on CPU (Mac):** ACT 1-cam & 2-cam build/forward/backward/predict/predict_chunk; KL
annealing (`loss == l1` at step 0) + free-bits + `_kl_step` increment; smoke tests (act, diffusion)
end-to-end; `test_inference` all-pass; the probe runs end-to-end. DINOv2's `torch.hub` load itself
was **not** triggered locally (avoided the download).

**Left for the Linux/GPU box:**
1. Train a few epochs on a dino config → confirm **KL stabilizes > 0** (not `0.21 → 0.0001`).
2. `shortcut_probe.py --ckpt <best>` → confirm the model is **less state-dominant than 2.2×**;
   use `--checkpoints …` to select.
3. Point `dataset.root` at the 2-camera dataset (re-convert with
   `--cameras wrist,scene --extract-stride 5` if needed).

**Still the real-world root fix (out of this branch's scope):** re-record **fixed-home + varied
marker** data — no code change fully removes the proprioceptive shortcut if the data still lets the
joint state predict the grasp (file 4 §4.2).
