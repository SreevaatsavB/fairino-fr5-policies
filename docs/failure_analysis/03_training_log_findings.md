# 3. Training-Log Findings — what the training tells us (and the bug it hides)

[File 1](01_symptoms_and_evidence.md) and [file 2](02_failure_modes_explained.md) explained the
*deploy* behaviour. This file looks at the **training logs** themselves
(`experiment_data/training_logs/`) and finds the training-side causes — most importantly **one
concrete bug** (the CVAE switched itself off) and several "this metric is lying to you" traps.

Logs analysed (5 ACT runs + 1 Octo run):

| Log | Config |
|---|---|
| `train_dino_frozen.out` | DINOv2 fully frozen, no dropout |
| `train_dino_gripper_dropout.out` | + gripper dropout 0.5 |
| `train_dino_partial_unfreeze.out` | DINOv2 last block fine-tuned + gripper dropout |
| `train_dino_proprio_dropout.out` | + proprio dropout 0.5 + gripper dropout 0.5 (**this is the deployed model**) |
| `train_act_abs_eef_resume.log` | ACT with absolute-EEF actions (resumed run) |
| `train_octo_2cam.log` | Octo base-1.5, 2-camera (different model — skimmed) |

---

## 3.1 The headline: every ACT run trained "successfully" and still fails

There were **no crashes, no NaNs, no exploding losses, no obvious instability** in any ACT run.
The losses go down smoothly and early-stopping triggers normally. **This is the trap:** the
training looks healthy, so nothing screams "broken" — but the model it produces is object-blind.
The problems are **systematic bias**, not numerical failure. You have to know what to look for.

---

## 3.2 🔴 THE BIG ONE: CVAE posterior collapse (the latent is dead)

This is the most important training finding and it directly explains the **mode averaging**
(file 2 §2.3).

### What you should see vs what happened

ACT's loss has two parts: an **L1 reconstruction** term (predict the action) and a **KL** term
(keep the CVAE latent `z` informative). A healthy CVAE keeps the KL at some **non-trivial,
stable** value — that means `z` is actually carrying "which demo style" information.

In **every** ACT run, the KL term **collapses to ~zero**:

```
KL over epochs (representative, train_dino_frozen.out):
  ep1   0.2107
  ep2   0.0509
  ep3   0.0171
  ep5   0.0054
  ep10  0.0012
  ep15  0.0004
  ep20  0.0002
  ep30+ 0.0001   ← essentially zero (the partial-unfreeze run prints 0.0000)
```

Halved every 1–2 epochs, effectively dead by epoch ~15–20. With `kl_weight = 10`, the KL's
contribution to the loss at that point is `10 × 0.0001 = 0.001` — negligible next to an L1 of
~0.08–0.15.

### Why this is the whole ballgame

**Posterior collapse** = the CVAE latent `z` is being ignored. The encoder maps everything to
the prior, the decoder learns to produce the action **without using `z`**. When that happens,
ACT is no longer "a CVAE that commits to one mode" — it is just a **plain regression network**.
And a plain regression network, by the math in file 2 §2.3, outputs the **average** action →
the object-blind mean reach and the mid-air grasp.

> In other words: the one architectural feature that was supposed to stop mode-averaging
> **turned itself off during training**, in every single run, and nobody noticed because the
> overall loss still went down. This is a fixable bug, not a fundamental limitation.

**Likely cause:** `kl_weight = 10` is high, which (combined with a powerful decoder) is a classic
recipe for posterior collapse — the optimiser finds it cheaper to drive KL to zero than to use
the latent. Fixes (KL annealing, lower weight, free-bits) are in
[file 4 §4.1](04_root_cause_and_fixes.md).

---

## 3.3 🟠 The validation loss is lying to you (anti-correlation with deploy success)

This one is operationally dangerous because it means **your model-selection criterion is wrong**.

| Run | Best val_l1 | Deployed? | Deploy result |
|---|---|---|---|
| `dino_frozen` | **0.1444 (lowest)** | not the deployed one | — |
| `gripper_dropout` | 0.1471 | — | — |
| `partial_unfreeze` | 0.1485 | — | — |
| `abs_eef_resume` | 0.1586 | — | — |
| `proprio_dropout` | **0.1738 (highest)** | **yes (epoch 19)** | the failure you reported |

The model you actually deployed had the **worst** validation loss of the bunch, and it fails the
same way the "best val loss" model would. **Validation loss does not predict deploy success here
— if anything it is inverted.**

**Why:** a *lower* val loss can be achieved by fitting the training distribution's **mean pose**
more tightly. Since the mean pose *is* the (object-blind) failure behaviour, "better val loss"
can literally mean "better at being object-blind." No threshold on val_l1 can distinguish "uses
vision and generalises" from "predicts the average well."

**Consequence:** stop picking checkpoints by val_l1. Pick them with a **causal-intervention probe**
(does the action follow the image or the state?) or a **scored real/sim rollout**. See
[file 4 §4.5](04_root_cause_and_fixes.md).

---

## 3.4 🟠 The val-loss plateau is the "object-blind ceiling"

In every run, **train_l1 keeps dropping** (to ~0.05) while **val_l1 plateaus** at ~0.14–0.17 and
basically stops improving:

```
train_dino_frozen:  val_l1  0.220 → 0.149 (ep8) → 0.147 (ep15) → 0.1444 (ep28) → 0.149 (ep40)
                    train_l1 0.269 → ............................. → 0.059 (ep40)
```

- The **persistent gap** (val − train ≈ 0.077–0.105 across runs) on a 72–84M-parameter model
  trained on ~86 episodes is **memorisation**: the network fits the training trajectories but
  does not learn the visually-conditioned task.
- The **plateau height** (~0.14–0.17) barely changes across *every* config — frozen, fine-tuned,
  proprio-dropout, gripper-dropout. That stable ceiling is the residual error of the
  **object-blind mean predictor**. The model cannot get under it because getting under it would
  require *using vision to localise the marker*, which it has learned not to do.

### A subtle but important point about "low loss looks good"

train_l1 ≈ 0.05 *looks* excellent. But for **absolute** 7-D joint/gripper actions, "predict ≈ the
current pose / the trajectory mean" already yields a low L1 — so a low number here is consistent
with the **idle-action / predict-current-pose shortcut** (file 2 §2.1), not with task competence.
Low L1 is necessary but nowhere near sufficient.

---

## 3.5 🟡 Did the dropout regularisers actually do anything?

You added two regularisers to fight the shortcut. The logs show whether they "bit":

| Regulariser | Epoch-1 train_l1 vs frozen (0.269) | Verdict |
|---|---|---|
| **Proprio dropout 0.5** | **0.309** (~13% higher) | **Worked as intended** — masking the joint state genuinely made the task harder, which is exactly what you want (it stops the policy from leaning on state for those samples). It also produced the highest, most *honest* val_l1 (0.1738) and the earliest stop (ep31), i.e. it couldn't overfit as deeply. |
| **Gripper dropout 0.5** | **0.274** (~2% higher) | **Did almost nothing** — masking the gripper dimension barely changed the loss, because the model was barely using the gripper signal in the first place. Dropping a signal the model ignores has no regularising effect. |

So proprio dropout is pulling in the right direction (keep it), but it is **not sufficient alone**
because (a) the data still permits the shortcut and (b) the CVAE collapse reintroduced mean
behaviour. Gripper dropout is mostly a no-op for the loss — the gripper problem needs a
**contact/force signal**, not dimension-masking (file 4 §4.4).

---

## 3.6 🟡 Other weird / notable things in the logs

| Finding | Detail | Why it matters |
|---|---|---|
| **Corrupted frames** | 14 frames permanently replaced with zeros (same indices every epoch) across 11 train + 3 val episodes — e.g. `ep82/idx1010`, `ep44/idx945`, … | ~14% of episodes have a zeroed visual frame at a fixed timestep — minor but real data poisoning; worth fixing the converter / re-extracting those frames. |
| **"Frozen" vs "partial_unfreeze" not capacity-matched** | frozen = 72.7M trainable; partial_unfreeze = 74.5M (the extra 1.8M = the unfrozen DINOv2 last block at 0.1× LR) | The comparison between those two runs is muddied by a capacity difference — keep that in mind when reading their val numbers. |
| **abs_eef run** | resumed at epoch 5; smaller dataset (70 train / 17 val vs 86/21); best val_l1 0.1586 @ ep36; KL already collapsed by ep5 | Absolute-EEF didn't help and the log is missing epochs 1–5; its higher val is partly the smaller dataset. |
| **Octo 2-cam run** | trained 15k steps cleanly; **only** issue is a `TensorFlow not built for compute capability 12.0 → JIT-from-PTX` warning (slow ~9 s first step) and benign `CropAndResize PredictCost` messages | No training instability; the GPU is newer than the prebuilt TF supports (Ada/Blackwell-class), so TF JIT-compiles kernels at startup. Cosmetic/perf only. |
| **No step-loss for Octo** | WandB in offline mode → per-step losses are in WandB, not the log | Can't assess Octo's loss curve from the log alone. |

---

## 3.7 Cross-run comparison table

| Config | Best val_l1 | Best epoch | Final train_l1 | KL @ best ep | Early stop @ | val − train gap |
|---|---|---|---|---|---|---|
| dino_frozen | **0.1444** | 28 | 0.059 (ep40) | ~0.0001 | 40 | ~0.077 |
| gripper_dropout | 0.1471 | 40 | 0.054 (ep52) | ~0.0001 | 52 | ~0.093 |
| partial_unfreeze | 0.1485 | 53 | 0.049 (ep65) | 0.0000 | 65 | ~0.100 |
| abs_eef_resume | 0.1586 | 36 | 0.053 (ep48) | ~0.0001 | 48 | ~0.105 |
| **proprio_dropout** (deployed) | **0.1738** | 19 | 0.067 (ep31) | ~0.0001 | 31 | ~0.104 |

Read this table as: *every* run collapsed the KL, *every* run has a big memorisation gap, the
val numbers are all clustered around the object-blind ceiling, and the deployed run is the one
with the highest (worst) val loss — confirming val loss is not the right selector.

---

## 3.8 The four training-side verdicts (direct answers)

1. **Is there CVAE posterior collapse?** **Yes — completely, in every run.** KL 0.21 → ~0.0001.
   ACT was running as plain regression. (→ mode averaging, file 2 §2.3.) **This is the top bug.**
2. **Is the L1 suspiciously low / idle-action-shaped?** **Yes.** train_l1 ~0.05 with a val
   plateau ~0.15 is consistent with "predict ≈ current pose," not task competence.
3. **Did the dropout regularisers change the loss as expected?** **Proprio dropout: yes**
   (+13% train loss, the right direction). **Gripper dropout: no** (+2%, basically a no-op).
4. **Does best val_l1 correlate with deploy success?** **No — it is inverted.** The deployed
   model had the worst val loss. Don't select on val loss.

Next: [`04_root_cause_and_fixes.md`](04_root_cause_and_fixes.md) turns all of this into a
prioritized action plan.
