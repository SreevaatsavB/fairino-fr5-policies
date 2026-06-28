# 4. Root Cause and Fixes — what to actually do

This is the "do something" file. It states the **root-cause chain** in one place, then gives a
**prioritized fix list**: what to change, why, how to do it concretely, and how to **verify** it
worked. Fixes are ordered by *impact per unit effort* given the evidence — do them top to bottom.

If you read only one thing: **fix the CVAE collapse (§4.1) and fix the data (§4.2) first.**
Everything else is secondary until those two are done.

---

## 4.0 The root-cause chain, in one paragraph

The demonstrations let the policy predict the action from the **joint state** instead of the
**cameras** (proprioceptive shortcut), and ACT's defense against averaging — the **CVAE latent** —
**collapsed during training** (KL → 0), so the policy became a plain regressor that outputs the
**average** demonstrated motion. That average is **object-blind** (reaches a fixed phantom spot),
**freezes** near the grasp because pause frames taught it "stay still there" (idle-action), and
**closes the gripper on air** because the close is timed, not contact-triggered, and the binary
gripper state can't tell a real grasp from a fake one. Once it fakes the grasp it is
**out-of-distribution** and never recovers to the place phase. Fix the *data* and the *CVAE* and
most of this disappears; fix the *gripper signal* and the *idle frames* to finish the job.

```
ROOT CAUSES (fix these)                         SYMPTOMS (these then go away)
─────────────────────────                       ──────────────────────────────
data lets state predict the grasp          ─►   object-blind reach to a fixed spot
CVAE posterior collapse (KL→0)             ─►   mean / averaged motion
absolute-joint actions + idle frames       ─►   freeze near the grasp
gripper close is timed, state is binary    ─►   mid-air grasp + phantom transport
all of the above → off-distribution        ─►   never reaches the bowl
```

---

## 4.1 FIX #1 — Stop the CVAE posterior collapse (training bug; highest leverage)

**Why first:** it is a concrete, certain bug (KL → 0 in every run, [file 3 §3.2]); fixing it
re-enables the one mechanism that prevents mode-averaging; and it is a few-line change.

**What to do (any/all of):**
- **Lower `kl_weight`** from `10` to roughly `0.1–1.0`. A weight of 10 over-penalises the latent
  and is a classic collapse trigger.
- **KL annealing / warm-up:** start `kl_weight` near 0 and ramp it up over the first few thousand
  steps, so the decoder learns to *use* `z` before it is pressured to shrink it.
- **Free bits:** floor the KL per latent dimension (e.g. `max(KL, 0.1)`) so the optimiser cannot
  drive it to exactly zero.

**How to verify it worked:** plot the KL over training — it should **stabilise at a non-trivial
value** (not crater to ~0.0001). Then do a quick check that the latent matters: with two
different `z` samples the predicted action chunk should differ noticeably for the same
observation. If KL stays healthy and `z` changes the output, the CVAE is alive again.

> Reference: ACT relies on the CVAE (removing it drops success 35%→2%, Zhao et al. 2023); posterior
> collapse is a well-known VAE failure with high KL weight (Bowman et al. 2016, "Generating
> Sentences from a Continuous Space" — the origin of KL annealing).

---

## 4.2 FIX #2 — Fix the DATA so vision *has* to be used (highest leverage, more effort)

**Why:** the proprioceptive shortcut (file 2 §2.2) is a property of the **data**, not the model.
No architecture change fully fixes it if the data still lets the joint state predict the grasp.

**What to do:**
- **Fixed home + widely varied marker.** Start every demo from the *same* home pose, and place
  the marker across the **full reachable workspace** (a grid is ideal — e.g. 4×4 or 5×5 cells,
  several demos per cell). Now the start state carries **zero** information about where to grasp,
  so the only way to lower the loss is to **look at the camera**. This is the single most
  important data change.
- **More demonstrations.** ~86 episodes is small; target **100–150+** with the varied-marker
  protocol. Below ~100 the cheap state shortcut almost always wins.
- **Vary the bowl/target position too**, so the place phase also has to be vision-conditioned.
- **Filter idle/pause frames** (this also helps the freeze, §4.3).

**How to verify:** after retraining, run the **causal-intervention probe** (you already have one —
the `diag_ablation` / `predict_with_diag` instrumentation): blank the cameras vs. perturb the
state and compare how much the action changes. Today proprio is **2.2× > vision**; the target is
**image-dominant** (the action should change *more* when you blank the cameras than when you
perturb the state). That number is your success metric for this fix.

> References: de Haan et al. 2019 (causal confusion); Zhao et al. 2025 (state-free, 0%→85% from
> exactly this kind of change).

---

## 4.3 FIX #3 — Kill the freeze: remove idle frames + use relative actions

**Why:** the freeze is the idle-action problem (file 2 §2.1), amplified by the absolute-joint
action space.

**What to do:**
- **Filter idle/pause frames** from the dataset before training: drop frames where the
  per-step joint motion is below a small threshold (`‖Δq‖ < ε`) or where the action repeats the
  previous one for N frames. The Diffusion Policy paper shows this is *the* fix for "policy gets
  stuck." Be careful to keep a few low-motion frames around genuine grasp moments so the policy
  still learns to slow down — just don't let pauses dominate.
- **Switch to relative actions** (delta joint or, better, **delta end-effector**). In delta space
  "do nothing" is the **zero vector**, which is unambiguous and easy to learn *not* to over-predict;
  in absolute space "do nothing" = "predict my current pose," which is the cheap shortcut that
  feeds the freeze. (Your repo already has the `--action-space delta_eef` path; this is one of its
  benefits. Note delta-EEF also needs the IK deploy path — see `docs/action_spaces.md`.)

**How to verify:** in a rollout, the per-step commanded motion near the grasp should **not**
collapse to ~0; and `mean |chunk0 − state|` (from `diag_offline`) should no longer be ~0.84° of
"stay put" when the task wants motion.

> References: Chi et al. 2023 (remove idle actions); Wen et al. 2020 (copycat — relative/short
> history breaks the shortcut).

---

## 4.4 FIX #4 — Make the grasp contact-aware (stop the mid-air close + phantom transport)

**Why:** the gripper closes on timing and the binary state can't tell a real grasp from a fake
one (file 2 §2.4).

**What to do:**
- **Gate the transport phase on an actual grasp signal.** You have `gripper_norm` in the 7-D
  state already; even better, read the **gripper width / force** at deploy. After a close command,
  check: did the gripper *fail to fully close* (object present, width stays open) or *close all the
  way* (likely empty)? Only proceed to "move to bowl" if the grasp looks real; otherwise
  **re-attempt** the approach. This is a deploy-side closed-loop check, independent of the policy.
- **Add a contact/force or pseudo-tactile observation** so the policy itself can perceive
  "holding vs empty" (the three states open-empty / closed-on-object / closed-on-air become
  distinguishable). The cited paper does this with the force-gripper's own joint angle — no extra
  hardware.
- **Collect more grasp-phase data** (close-up, gripper-near-object views) so the contact moment is
  in-distribution.

**How to verify:** in rollouts the gripper should only commit to "closed → transport" when a grasp
is actually detected; mid-air closes should trigger a retry, not a phantom carry.

> Reference: *Disambiguate Gripper State in Grasp-Based Tasks* (arXiv:2503.23835).

---

## 4.5 FIX #5 — Stop selecting models by validation loss

**Why:** val_l1 is anti-correlated with deploy success here (file 3 §3.3) — the deployed model had
the *worst* val loss.

**What to do:**
- **Select checkpoints by the causal-intervention probe** (image-vs-state dependence) and/or a
  **scored rollout** (sim or real), not by val_l1.
- Track the **vision-dependence ratio** as a first-class metric during training (you already
  compute it). A model isn't "better" because its val loss is lower; it's better because it
  *uses the cameras*.

**How to verify:** the metric you optimise should move with real rollout success. If lowering
val_l1 doesn't improve rollouts, you're tuning the wrong number.

---

## 4.6 FIX #6 (structural, if the above isn't enough) — change the policy class

**Why:** ACT-with-collapsed-CVAE mode-averages; if you can't keep the CVAE healthy, use a policy
that handles multimodality natively.

**Options:**
- **Diffusion Policy** — denoises to a *single* mode instead of averaging; the Diffusion Policy
  paper shows it doesn't even need idle-frame removal to avoid getting stuck.
- **Octo (finetuned)** — you already have this wired (`policies/octo/`). It is vision+language,
  no proprioceptive state input by default — so it **structurally avoids the proprioceptive
  shortcut**, and it brings an 800k-trajectory visual prior, which is exactly what a ~86-episode
  dataset lacks. Strong candidate to compare against the fixed ACT.

> References: Chi et al. 2023 (Diffusion Policy); Octo docs in this repo (`docs/octo*.md`).

---

## 4.7 The priority checklist (tear-off version)

```
[ ] 1. CVAE: lower kl_weight to ~0.1–1.0 + KL anneal/free-bits; verify KL stays > 0 and z matters
[ ] 2. DATA: re-record fixed-home + varied-marker (grid), 100–150+ demos, vary bowl too
[ ] 3. FREEZE: filter idle/pause frames; move to relative (delta-EEF) actions
[ ] 4. GRASP: deploy-side grasp-success gate (gripper width/force) + add contact obs; more grasp data
[ ] 5. SELECTION: pick checkpoints by the vision-vs-state probe / scored rollout, NOT val_l1
[ ] 6. (if needed) try Diffusion Policy or finetuned Octo as a multimodal/state-free alternative
[ ] re-run the causal-intervention probe after each change — target: image-dominant (not 2.2x state)
```

### Expected payoff per fix (rough, evidence-based)

| Fix | Attacks | Expected effect |
|---|---|---|
| #1 CVAE | mode averaging | restores ACT's anti-averaging mechanism; less "mean" behaviour |
| #2 Data | proprioceptive shortcut | the big one — makes the reach actually track the marker |
| #3 Idle+relative | freeze | arm keeps moving through the grasp instead of stalling |
| #4 Grasp gate | mid-air grasp + no-bowl | real grasps; retries instead of phantom transport; reaches bowl |
| #5 Selection | choosing a bad model | you stop shipping the object-blind checkpoint |
| #6 Policy class | residual averaging / small data | structural robustness if ACT still struggles |

**Validation north-star:** the single number that summarises whether you've fixed the core
problem is the **vision-vs-state dependence ratio** from your own probe. It is **2.2× toward
state** today. When it flips to **image-dominant**, the object-blindness — and most of the cascade
— is gone.

See [`05_glossary_and_references.md`](05_glossary_and_references.md) for definitions and all paper
links.
