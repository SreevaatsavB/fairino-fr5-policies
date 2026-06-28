# 1. Symptoms and Evidence — what the robot actually did

This file is the **crime scene**: the precise, measured behaviour of the policy on the real
FR5, pulled directly from your deploy logs (`experiment_data/inference_logs/`) and the two
diagnostic NumPy files. No interpretation yet — just *what happened* and *the numbers that
prove it*. The explanations of *why* are in
[`02_failure_modes_explained.md`](02_failure_modes_explained.md).

---

## 1.1 The reported behaviour (operator's words, made precise)

> "Almost every time, the robot reaches near the marker, then freezes, then goes to the end
> state. It does not pick the marker — or if it does, it grasps mid-air and stops. We never
> saw it move toward the bowl."

Broken into four distinct, separately-measurable events:

1. **Approach** — the arm leaves home and sweeps toward roughly where the marker usually is.
2. **Freeze** — partway there (near the grasp), motion collapses; the arm essentially stops.
3. **Phantom grasp** — the gripper closes while the hand is still in the air (nothing in it).
4. **No transport** — the arm either stays frozen or retracts a little, but the
   "carry the marker to the bowl" phase never happens.

Every one of these is visible in the logs. Here is the evidence, event by event.

---

## 1.2 Event 2 — the FREEZE (this is the most clear-cut)

**What "freeze" means numerically:** the commanded joint angles stop changing, and the
end-effector (EEF / TCP) position stops moving, for hundreds of control steps.

### Direct evidence

| Log file | What it shows |
|---|---|
| `deploy_grip2.out` | The arm moves slightly from home, then **EEF displacement = `[0.0, 0.0, 0.0]` mm from step 134 through step 570** — i.e. **436 consecutive control steps with zero motion**. The frozen pose is only **1.48° (mean) from the home pose**. |
| `deploy_bowl.out` | After step ~663 the TCP moves **< 0.6 mm over 1329 steps** (402.6 → 403.2 mm). Joint motion in the frozen phase is ~**0.021°/step** vs ~0.056°/step while still active — but the *tool tip* is completely still. |
| `deploy_prop.out` | Reaches ~393.9 mm EEF by step ~472, then joint positions change **< 5° total over the next 1517 steps**. |
| `intervention.out` | Under policy control the arm only ever moved **2–4 mm from rest** before stalling; a human took over and moved **129 mm** in the same setup in under 2 minutes. The policy is **~30–60× weaker at making progress than a human in the identical scene.** |

### The offline fingerprint of the freeze

From `diag_offline.npz` (90 inference calls on *training-distribution* observations):

- `mean |chunk0 − current_state| = **0.84°**` — the **very first action** the policy predicts
  is, on average, **less than one degree** away from where the arm already is. The policy's
  default prediction is *"stay almost exactly where you are."*
- The full 100-step chunk only spans `mean |chunk[99] − chunk[0]| = 5.75°` per joint — it
  *plans* small motions, with **tiny first steps**.

**Plain-language reading:** the model has learned that the correct next action is "barely
move." That is fine while it is still approaching (lots of barely-moves add up), but near the
grasp the barely-moves average out to **nothing**, and the arm stops. This is the signature of
the **idle-action problem** (file 2, §2.1).

---

## 1.3 Event 3 — the PHANTOM (mid-air) GRASP

**What it means numerically:** the gripper command crosses its "close" threshold while the
end-effector is still far from the marker / table — i.e. the hand closes on empty space.

### Direct evidence

| Log file | Gripper close event | EEF distance from start at that moment |
|---|---|---|
| `deploy_ft.out` | `[GRIPPER] CLOSED` at step ~553 | **386.9 mm** — then the arm **retracts to 312.6 mm** (as if carrying something) |
| `deploy_grip4.out` | `[GRIPPER] CLOSED` at step ~526 | **363.6 mm** — then retracts to 285.1 mm |
| `deploy_bowl.out` | gripper jumps 0.02 → 0.48 at step 663 | **402.6 mm**, no contact; then oscillates 0.43–0.63 for 1329 steps (never commits) |
| `deploy_dino.out` / `deploy_dino2.out` | `[GRIPPER] CLOSED` at **step 2** | **27 mm / 0 mm** — closes *immediately*, before any reach. (These runs had **no gripper dropout** in training.) |

Two distinct sub-patterns:

- **Without gripper dropout** (`deploy_dino*`): the gripper closes at **step 2**, basically at
  the start pose. The model learned "always start closed" — it never even tries to localise.
- **With gripper dropout** (`deploy_ft`, `deploy_grip4`, `deploy_bowl`): the gripper closes
  **mid-reach** (360–403 mm out), then the arm **switches into a "transport" motion** (it
  retracts) — behaving exactly as if it had successfully grabbed the marker, even though it
  grabbed air.

**Plain-language reading:** the gripper close is fired by **trajectory timing / arm pose**, not
by actual contact with the object. And because the policy's only notion of "am I holding
something?" is the binary gripper state (closed = "I have it"), once it closes it *believes* it
succeeded and moves to the next phase. This is **gripper-state ambiguity** (file 2, §2.4).

---

## 1.4 Event 1 + 4 — APPROACH to a fixed spot, and NO bowl

**What it means:** the arm reaches toward the *same* region every rollout, regardless of where
the marker actually is, and never proceeds to the place phase.

### Direct evidence

- The frozen/grasp poses across `deploy_bowl`, `deploy_prop`, `deploy_ft`, `deploy_grip4` land
  in a **similar EEF band (~360–403 mm out)** — a *fixed average reach*, not one that tracks
  the marker.
- **Your own causal-intervention ablation** (the reason the `_prop` config exists, quoted in
  `config.joint_dino_prop.yaml`): *"proprio influenced the action 2.2× more than vision."* The
  policy's output depends **2.2× more on the joint state than on the cameras** — so it reaches
  the same place no matter what the camera sees.
- In **every** log, after the phantom grasp the EEF magnitude **never decreases toward a
  different (bowl) location**; the task simply ends in a freeze or a small retract.

**Plain-language reading:** the policy is **object-blind**. It reproduces the average reach and
the average grasp timing, ignoring the actual marker position — the **proprioceptive shortcut /
mode averaging** (file 2, §2.2 and §2.3). And since the grasp never really succeeds, the
later "move to bowl" behaviour — which in the data *always* begins from "object in hand" — is
never triggered (**covariate shift**, file 2, §2.5).

---

## 1.5 The diagnostic `.npz` files (the quantitative core)

These two files were produced by the policy's own instrumentation
(`predict_with_diag` / `predict_chunk` on the `act` branch). They are the most precise evidence.

### `diag_offline.npz` — 90 open-loop predictions on in-distribution data

| Array | Shape | Meaning |
|---|---|---|
| `chunks` | (90, 100, 7) | full predicted 100-step action chunks |
| `chunk0` | (90, 7) | first action of each chunk |
| `ensembled` | (90, 7) | temporally-ensembled output (what gets executed) |
| `state` | (90, 7) | actual joint state at prediction time |
| `latency_ms` | (90,) | mean **13.1 ms**, max 137 ms (well under the 33 ms budget → timing is NOT the problem) |

Key numbers:

- `mean |chunk0 − state| = **0.84°**` → first action ≈ "stay put" (the freeze fingerprint).
- `mean |ensembled − chunk0| = **4.87°**` (max 16.4°) → temporal ensembling is *adding* motion,
  not cancelling it (it blends older chunks' future steps).
- Direction agreement between `chunk0` and `ensembled` ≈ **58.7%** → barely above chance: the
  successive predictions point in **incoherent directions** ~41% of the time.

### `diag_receding.npz` — 346 steps of pure receding-horizon (temporal ensembling OFF)

| Finding | Number | Meaning |
|---|---|---|
| Robot tracking error | `mean |executed − actual joints| = **0.26°**` | the hardware tracks commands accurately — **not a robot problem** |
| Per-step motion | `mean |Δjoint| = **0.17°/step**` | without ensembling the arm barely moves |
| **Sign-flip rate** | **68%** | each fresh `chunk0` predicts the **opposite direction** from the previous one 68% of the time → near-pure oscillation/jitter |
| Total joint range over 345 steps | J1 7.2°, J4 13.6°, others < 5° | the arm essentially **never leaves home** without ensembling |

**Important nuance about temporal ensembling:** turning it **off** made things *worse*
(`deploy_receding.out`: the arm oscillated within 33 mm and never progressed). So temporal
ensembling is **not** the villain — it is actually the only thing letting the arm travel at
all. The 68% sign-flip rate shows the *underlying predictions* are incoherent; ensembling
smooths them into forward motion, but it cannot fix a policy that fundamentally wants to
stop. (This rules out "temporal-ensembling cancellation" as the root cause.)

---

## 1.6 Offline looks OK, online is broken (the covariate-shift tell)

A crucial detail: **all the checkpoints have similar, reasonable validation loss**
(val_l1 ≈ 0.144–0.174 — see file 3), and the **offline** predictions in `diag_offline.npz`
look like sane small motions. Yet **online** the rollouts consistently fail.

That gap — *offline metrics fine, closed-loop rollout broken* — is the classic signature of
**compounding error / covariate shift** (file 2, §2.5): the policy is only ever tested,
offline, on states the experts visited; online it drifts into states it never saw and has no
good action for, so it regresses to "do almost nothing."

---

## 1.7 Things that are NOT the problem (ruled out by the data)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Robot hardware / tracking | **Not it** | tracking error 0.26°; commands are followed accurately |
| Control-rate / latency | **Not it** | inference 12–13 ms, far under the 33 ms (30 Hz) budget |
| Numerical instability (NaN/Inf) | **Not it** | no NaNs/Infs in any log or array |
| Temporal-ensembling cancellation | **Not the root cause** | turning it off made motion *worse* (33 mm jitter) |
| A single bad checkpoint | **Not it** | the failure reproduces across epochs 19/28/40/53 and across configs |
| `deploy_receding.out` crash | **Setup issue, not policy** | `OSError: GetActualJointPosDegree returned 14` = CNDE disconnect from a suspended `teleop.py` sharing the port |

---

## 1.8 Evidence → phenomenon map (forward reference)

| Measured fact | Points to | Detailed in |
|---|---|---|
| `chunk0 ≈ state` (0.84°); 436-step 0 mm freeze | **Idle-action problem** | file 2 §2.1 |
| 68% sign-flip; near-zero net motion | **Copycat / inertia** (same family) | file 2 §2.1 |
| proprio 2.2× > vision; same reach regardless of marker | **Proprioceptive shortcut / causal confusion** | file 2 §2.2 |
| object-blind "average" grasp; CVAE off in training | **Mode averaging / regression to mean** | file 2 §2.3 |
| gripper closes at 360–403 mm, then "transports" nothing | **Gripper-state ambiguity** | file 2 §2.4 |
| offline OK, online broken; no recovery after phantom grasp | **Covariate shift / compounding error** | file 2 §2.5 |

Next: [`02_failure_modes_explained.md`](02_failure_modes_explained.md) explains each of these
from first principles.
