# FR5 ACT (DINOv2) — Failure Analysis

This folder is a complete, from-the-ground-up explanation of **why the trained ACT +
DINOv2 policy failed on the real FR5 robot**, and what to do about it. It is written so
that you can hand it to anyone on the team and they will understand the problem fully —
no prior context needed.

## The one-sentence answer

> The policy learned to reproduce the **average demonstrated trajectory** and to **ignore
> the cameras**, so on the robot it reaches toward the *average* marker location, **freezes**
> near the grasp, **closes the gripper in mid-air**, and — because it never actually grasps
> anything — **never starts the move-to-bowl phase**.

That single sentence is the product of **three well-known imitation-learning failure modes
stacking on top of each other**, plus **one concrete training bug** (the CVAE turned itself
off) that made the whole thing worse. Each is documented in detail in the files below.

## What was analysed

- **The model:** lerobot `ACTPolicy` (CVAE + transformer, chunk 100, temporal ensembling
  0.01) with a **frozen DINOv2 ViT-S/14** vision backbone, **two cameras** (wrist + scene),
  7-D action (6 joints + gripper). Some runs used **proprioception dropout 0.5** and
  **gripper dropout 0.5**. This is the `act` branch of `vivek-kanjarla/fairino-fr5-policies`.
- **The data:** training logs (5 ACT runs + 1 Octo run) and inference/deploy logs (12 robot
  rollouts + 2 diagnostic `.npz` files) that you uploaded — see `experiment_data/`.
- **The task:** pick up a marker and place it in a bowl.

## How to read this folder (in order)

| # | File | What it covers |
|---|---|---|
| 1 | [`01_symptoms_and_evidence.md`](01_symptoms_and_evidence.md) | Exactly what the robot did, mapped to the **hard numbers** from your deploy logs and diagnostic files. The "crime scene." |
| 2 | [`02_failure_modes_explained.md`](02_failure_modes_explained.md) | Every named phenomenon explained from scratch — idle-action freeze, copycat/inertia, mode averaging, proprioceptive shortcut, gripper-state ambiguity, covariate shift — with the mechanism, the citation, and how it produced *your* symptom. The deep dive. |
| 3 | [`03_training_log_findings.md`](03_training_log_findings.md) | What the **training** logs reveal: the **CVAE posterior collapse** (the big bug), the val-loss anti-correlation, the loss plateau, corrupted frames, and a per-run comparison. |
| 4 | [`04_root_cause_and_fixes.md`](04_root_cause_and_fixes.md) | The full causal chain, the root causes ranked, and a **prioritized, actionable fix list** with how to validate each one. Start here when you want to *do* something. |
| 5 | [`05_glossary_and_references.md`](05_glossary_and_references.md) | Plain-language definitions of every term, and all the papers with links. |
| 6 | [`06_cvae_kl_deep_dive.md`](06_cvae_kl_deep_dive.md) | **The CVAE KL term explained from scratch** — what KL divergence is, the ELBO, the reparameterisation trick, and exactly why `kl_weight=10` caused **posterior collapse** (the top training bug). Read after file 3. |
| 7 | [`07_grasp_gate_requirements.md`](07_grasp_gate_requirements.md) | The deferred **grasp-success gate** — what sensor signal it needs (the FR5 gripper is command-only) and where it plugs into `deploy.py`. |
| 8 | [`08_the_fix_implementation.md`](08_the_fix_implementation.md) | **What we actually changed** (branch `fix/dino-act-cvae-probe`) — the CVAE fix (free-bits + annealing + lower weight, explained), the DINOv2 2-cam port, the vision-vs-state probe, the full change set, and what's verified vs left. |
| 9 | [`09_model_capacity_vs_overfitting.md`](09_model_capacity_vs_overfitting.md) | **"Is the model too small?" — No.** Why the large train/val gap is **overfitting** (data-bound), not under-capacity; the 32-dim latent is standard; scale *data* not params. |
| 10 | [`10_free_bits_explained.md`](10_free_bits_explained.md) | **Free-bits explained in detail** — the "tax-free allowance per latent dim" idea, the gradient view (ASCII curve), a worked example, the code, and why `λ=0.03` is kept gentle. |

## The 30-second version (if you read nothing else)

```
WHAT YOU SAW                     WHAT IT IS CALLED                 PROOF IN YOUR LOGS
────────────────────────────     ─────────────────────────────    ─────────────────────────────
reaches ~same spot every time →  proprioceptive shortcut /     →  your ablation: proprio 2.2x > vision
ignores where the marker is      causal confusion (de Haan'19)

grasps a "mean" location,     →  mode averaging / regression  →  CVAE KL collapsed to ~0 in training
object-blind, in mid-air         to the mean (Diffusion Policy)    => ACT became plain regression => mean

freezes near the marker       →  idle-action problem          →  deploy_grip2: 0.0 mm motion for 436 steps;
                                 (Chi et al. 2023) + copycat       chunk0 ≈ current pose (0.84 deg)

closes gripper on nothing,    →  gripper-state ambiguity      →  gripper CLOSED at 360-403 mm from start,
then "retracts to transport"     (arXiv 2503.23835)                then arm retracts as if carrying it

never reaches the bowl        →  compounding error /          →  after the phantom grasp it is out of
                                 covariate shift (Ross 2011)       distribution; no recovery behaviour
```

**The most important single fix:** the CVAE latent collapsed during training (KL → ~0),
which silently turned ACT into ordinary behavioral cloning — and ordinary BC mode-averages.
Fix that first (see file 4), then fix the data (fixed home + varied marker) and filter idle
frames. Details and the full priority list are in
[`04_root_cause_and_fixes.md`](04_root_cause_and_fixes.md).

> Note: the analysis is evidence-based — every claim in these files is tied either to a
> specific number in your logs (`experiment_data/…`) or to a named paper. Where something is
> inferred rather than directly measured, it says so.
