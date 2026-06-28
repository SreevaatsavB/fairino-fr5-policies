# 2. The Failure Modes Explained (from scratch)

This is the deep dive. Each section below is **one named phenomenon** from the imitation-
learning literature. For each you get: a plain-language explanation, the precise mechanism
(with the small amount of math that matters), the **canonical paper(s)**, and exactly **how it
produced your symptom** with the evidence from [file 1](01_symptoms_and_evidence.md).

Read the preamble first — it explains *why behavioral cloning is fragile in the first place*,
which is the common root of everything that follows.

---

## 2.0 Preamble — why behavioral cloning breaks at all

Your ACT policy is trained by **behavioral cloning (BC)**: you record human demonstrations as
`(observation, action)` pairs and train a network to output the action the human took. The
objective is just supervised learning:

```
minimize   E[ (π(observation) − action)² ]
```

That's it. There is **no notion of the goal, no feedback loop, and no understanding of cause
and effect** — only "given this input, copy that output." Three structural gaps follow, and
every failure mode below lives in one of them:

| Gap | Consequence | Failure modes |
|---|---|---|
| BC fits **correlations**, not causes | it can lean on inputs that merely correlate with the action (joint state, previous action) instead of the ones that *cause* it (the camera) | §2.1 copycat, §2.2 proprioceptive shortcut |
| BC minimises average error → predicts the **mean** | when several actions are valid it outputs their average, which can be invalid | §2.3 mode averaging |
| BC is only trained on **expert states** | at deploy it drifts into states it never saw and has no good action | §2.5 covariate shift |

ACT tries to patch the second gap with a **CVAE** (a latent "style" variable that lets it pick
*one* mode instead of averaging). **In your runs that patch failed** — the CVAE collapsed (see
[file 3](03_training_log_findings.md)) — so the mean-averaging behaviour came back in full.

---

## 2.1 The idle-action problem (and its cousin, the copycat/inertia problem)

**This is the FREEZE.** The single most-supported cause in your data.

### Plain language

Humans pause while teleoperating — they slow down to line up the grasp, they hesitate, they
hold still for a moment before closing the gripper. Those pauses get recorded as **frames where
the action is "stay still."** Because BC weights every frame equally, and "stay still" frames
cluster at the **pre-grasp pose**, the network learns a very confident rule: *"when you are near
the grasp pose, do nothing."* At deploy, the arm reaches that pose and the policy votes,
confidently, for **no motion** — it freezes.

### The mechanism, precisely

Two reinforcing effects:

1. **Idle frames are low-loss and over-represented.** Predicting "stay still" at a state where
   the demo *was* still has near-zero error, and there are many such frames, so the gradient
   pushes the policy hard toward "stop here." (Chi et al., the Diffusion Policy paper, name this
   the **"idle actions"** problem and report a single-step BC baseline got *stuck in 40% of
   real-world trials* until idle frames were explicitly removed.)
2. **The copycat / inertia shortcut.** Consecutive expert actions are almost identical (the arm
   barely moves between two 30 Hz frames). So the *easiest* function for the network to learn is
   "output ≈ the previous action." If the previous action was "slowing down," the policy keeps
   outputting "slowing down" — a self-perpetuating stop. Wen et al. (NeurIPS 2020) name this the
   **"copycat problem"** and prove it happens when (a) actions are temporally correlated and
   (b) the previous action is recoverable from the input — both true here.

There is a subtle but important interaction with your **absolute-joint action space**: the
policy predicts *absolute* target joint angles, not deltas. "Do nothing" in absolute space means
"predict my current joint angles" — which, on average, is exactly what the data shows near the
pause. So the absolute action space makes "predict ≈ current pose" the lowest-loss answer.

### How it produced YOUR symptom

- `diag_offline.npz`: `mean |chunk0 − current_state| = **0.84°**` — the policy's default first
  action is "essentially stay where you are." That is the idle-action rule, measured directly.
- `deploy_grip2.out`: **0.0 mm motion for 436 steps** near home — the arm reached a "stay still"
  basin and never left.
- The absolute-joint design (above) is *why* "stay" is so cheap to predict.

### Canonical papers
- Chi et al., *Diffusion Policy* (RSS 2023) — names "idle actions," prescribes removing them.
  `arXiv:2303.04137`
- Wen et al., *Fighting Copycat Agents in Behavioral Cloning from Observation Histories*
  (NeurIPS 2020). `arXiv:2010.14876`

---

## 2.2 The proprioceptive shortcut / causal confusion

**This is the OBJECT-BLINDNESS** — why it reaches the same spot regardless of the marker.

### Plain language

The policy gets two kinds of input: **the joint state** (6 clean numbers describing where the
arm is) and **the cameras** (two 224×224 images it must learn to interpret). The joint state is
trivially easy to use and, in your demos, it is *correlated* with where the grasp ends up
(because the arm always starts from the same home pose and follows similar paths). So the
network takes the lazy route: it predicts the action **from the joint state** and barely uses the
cameras. At deploy the arm always starts at the same home pose, so the joint state is always the
same → the policy always produces the same "average" reach → it **ignores where the marker
actually is**.

### The mechanism, precisely

This is a specific case of **causal confusion** (de Haan, Jayaraman, Levine, NeurIPS 2019): BC
learns `P(action | observation)`, which is about *correlation*. It does not learn
`P(action | do(observation))`, which is about *causation*. The cameras *cause* the correct reach
(they say where the marker is); the joint state merely *correlates* with it in your particular
dataset. BC cannot tell the difference, and the joint state is the cheaper signal, so it wins.

A profoundly counter-intuitive consequence proven in that paper: **adding more "helpful" state
can make generalisation worse**, because it gives the network an even easier shortcut to exploit.
Zhao et al. 2025 ("Do You Need Proprioceptive States in Visuomotor Policies?") confirm it on real
robots — *removing* proprioception entirely took pick-and-place spatial generalisation from
**0% → 85%**.

### How it produced YOUR symptom

- **Your own ablation is the proof:** the `config.joint_dino_prop.yaml` header states
  *"proprio influenced the action 2.2× more than vision."* That is a direct measurement of the
  shortcut — the output moves 2.2× more when you perturb the joint state than when you perturb
  the image.
- Across rollouts the grasp pose lands in the same ~360–403 mm band regardless of the scene —
  a fixed *average* reach, exactly what an object-blind, state-driven policy produces.
- You correctly tried to fight it with **proprioception dropout 0.5** (zero the joint state half
  the time during training, forcing vision to contribute). That is the right instinct — but on
  its own it was not enough, because (a) the data still lets the shortcut form and (b) the CVAE
  collapse (file 3) reintroduced mean behaviour. See fixes in [file 4](04_root_cause_and_fixes.md).

### Canonical papers
- de Haan, Jayaraman, Levine, *Causal Confusion in Imitation Learning* (NeurIPS 2019).
  `arXiv:1905.11979`
- Zhao et al., *Do You Need Proprioceptive States in Visuomotor Policies?* (2025).
  `arXiv:2509.18644`

---

## 2.3 Mode averaging / regression to the conditional mean

**This is WHY the grasp is a "mean" location** — and it is the failure mode the CVAE was supposed
to prevent.

### Plain language

Your demonstrations are **multimodal**: the marker is in different places, and a human reaches
slightly differently each time (different approach angle, different wrist twist, different exact
grasp point). For a given camera view there is therefore **no single "right" action** — there are
several. BC with a squared/absolute-error loss responds to "several right answers" by predicting
**their average**. The average of "reach left" and "reach right" is "reach to the middle" — a
spot where *no demonstration ever grasped*. So the arm reaches a phantom average location and
closes on nothing.

### The mechanism, precisely

It is a one-line statistical fact. The function that minimises mean-squared error is the
**conditional mean**:

```
π*(o) = argmin_π  E[(π(o) − a)²]  =  E[a | o]      (the average action for that observation)
```

So a plain regression BC policy is *mathematically guaranteed* to output the average of all
valid actions for an observation. If those actions form two clusters (two modes), the average
lands between them — often an invalid, do-nothing, or wrong-location action.

**ACT's intended cure — and why it failed here.** ACT adds a **CVAE**: a latent variable `z`
that encodes *which* demonstration "style/mode" is being used. During training the encoder infers
`z` from the action sequence; at inference `z = 0` (the prior mean) is used, which commits the
decoder to **one** mode instead of averaging all of them. The ACT paper shows this matters
enormously — removing the CVAE drops success from **35.3% → 2%** on fine-manipulation tasks.

**In your runs, the CVAE turned itself off.** [File 3](03_training_log_findings.md) shows the
KL term collapsed to ~0 in *every* run — a phenomenon called **posterior collapse**. When the
KL collapses, the latent carries no information; the decoder ignores `z`; and ACT degenerates
into exactly the plain regression that mode-averages. So the very mechanism meant to prevent
§2.3 was silently disabled. **This is the most actionable single finding in the whole analysis.**

### How it produced YOUR symptom

- Object-blind "average" reach + mid-air grasp = textbook mode averaging.
- The training logs show **CVAE posterior collapse in all 5 ACT runs** (KL: 0.21 → 0.0001),
  i.e. ACT was running as plain mean-regression. (See file 3 §3.2.)

### Canonical papers
- Chi et al., *Diffusion Policy* (RSS 2023) — the multimodality/averaging motivation.
  `arXiv:2303.04137`
- Zhao et al., *ACT / Learning Fine-Grained Bimanual Manipulation* (RSS 2023) — the CVAE and the
  35%→2% ablation. `arXiv:2304.13705`
- Florence et al., *Implicit Behavioral Cloning* (CoRL 2021). `arXiv:2109.00137`
- Shafiullah et al., *Behavior Transformers (BeT)* (NeurIPS 2022). `arXiv:2206.11251`

---

## 2.4 Gripper-state ambiguity (premature / phantom grasp)

**This is the MID-AIR GRASP and the "retract as if carrying something."**

### Plain language

The policy decides when to close the gripper based on **where it is in the trajectory** (arm
pose / timing), not on whether it is actually touching the marker — because nothing in its input
tells it about contact. So it closes at the "usual" moment. Worse, its *only* sense of "do I have
the object?" is the **binary gripper state** (closed = 1). A gripper closed on a marker and a
gripper closed on air produce the **same** observation (state = "closed"). So once it closes on
nothing, the policy *believes the grasp succeeded* and moves to the next phase — it starts the
"transport" motion (you see the arm retract) while holding nothing.

### The mechanism, precisely

Two coupled issues, formalised in *"Disambiguate Gripper State in Grasp-Based Tasks"*
(arXiv:2503.23835):

1. **The close command is a learned temporal shortcut.** "Close at ~60–80% of the reach" is a
   reliable pattern in the data, so BC learns it as a function of arm progress, not contact.
2. **The binary gripper observation is ambiguous.** Three physical situations —
   *open-empty*, *closed-on-object*, *closed-on-air* — collapse into **two** observable values
   (open / closed). The policy literally cannot perceive the difference between a successful and
   a failed grasp, so it "transitions directly to the post-grasp stage, outputting a pull/transport
   action despite not having secured the object" (their words; your logs show exactly this).

### How it produced YOUR symptom

- `deploy_ft.out` / `deploy_grip4.out`: `[GRIPPER] CLOSED` at **EEF 360–403 mm from start**
  (mid-air), then the arm **retracts** (387→313 mm, 364→285 mm) — the phantom "transport."
- `deploy_dino*.out` (no gripper dropout): closes at **step 2** — the temporal shortcut at its
  most extreme (it learned "start closed").
- `deploy_bowl.out`: after the mid-air half-close, the gripper command **oscillates 0.43–0.63
  for 1329 steps** — it never even commits, because the gripper dimension is high-variance and
  unanchored to any real contact signal.

### Canonical paper
- *Disambiguate Gripper State in Grasp-Based Tasks: Pseudo-Tactile as Feedback Enables Pure
  Simulation Learning* (2025). `arXiv:2503.23835`

---

## 2.5 Covariate shift / compounding error (why it never reaches the bowl)

**This is the LAST-MILE failure** — and why "offline looks fine, online is broken."

### Plain language

BC is trained only on the states a *successful expert* visits. At deploy, the policy makes small
errors, which move it into slightly different states, where it makes bigger errors, and so on —
errors **compound**. Once it does something the expert never did (close on air), it is in a state
that **appears nowhere in the training data**. The "move to the bowl" behaviour in your data
*always* begins from "object is in hand." With no object in hand and an out-of-distribution
state, the policy has no learned action for "what now?" — so it stalls or wanders, and the bowl
phase never begins.

### The mechanism, precisely

Ross & Bagnell (DAgger, 2011) proved the classic result: under BC the expected error grows like
**O(T²)** in the task horizon `T` (versus O(T) if you could correct it interactively). A long
multi-stage task (reach → grasp → transport → place) is exactly the worst case: an early-stage
error puts you off-distribution and every later step amplifies it. Two extra effects make the
grasp the apex of the problem:

- **Close-up OOD:** at grasp range the wrist camera sees a big, partly gripper-occluded view that
  is *under-represented* in the demos, so the policy is most uncertain exactly where it most
  needs to be precise.
- **The faulty offline metric:** validation loss is computed on expert states, so it cannot see
  the compounding online drift. That's why your checkpoints have fine val loss yet fail online.

### How it produced YOUR symptom

- Every rollout ends at the (failed) grasp — the transport/place phase never starts.
- Offline `diag_offline.npz` predictions look like sane small motions, and val loss is OK
  (0.144–0.174), yet the closed-loop rollouts consistently fail — the canonical offline-fine /
  online-broken signature of covariate shift.

### Canonical papers
- Ross, Gordon, Bagnell, *A Reduction of Imitation Learning … to No-Regret Online Learning*
  (DAgger, AISTATS 2011).
- Laskey et al., *DART: Noise Injection for Robust Imitation Learning* (CoRL 2017).
  `arXiv:1703.09327`

---

## 2.6 How the five stack into the behaviour you saw

None of these acts alone — they **chain**:

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ DATA: human pauses near grasp; arm starts at fixed home; marker varies a bit;  │
  │       multimodal reaches; gripper close keyed to timing                        │
  └───────────────┬─────────────────────────────────────────────────────────────┘
                  │ train ACT with BC; CVAE collapses (KL→0, file 3)
                  ▼
  §2.2 proprioceptive shortcut ──► policy keys on joint state, ignores cameras
                  │
  §2.3 mode averaging (CVAE off) ─► outputs the AVERAGE reach → a fixed phantom spot
                  │
  §2.1 idle-action freeze ────────► near the grasp pose, predicts "stay still" → FREEZE
                  │
  §2.4 gripper-state ambiguity ───► closes on air; believes it succeeded → phantom grasp
                  │
  §2.5 covariate shift ───────────► now off-distribution; no recovery → NEVER reaches bowl
```

The order matters for fixing it: the **proprioceptive shortcut + mode averaging** make it
object-blind (reach the wrong place), the **idle-action problem** freezes it there, the
**gripper ambiguity** fakes a grasp, and **covariate shift** prevents any recovery.

Continue to [`03_training_log_findings.md`](03_training_log_findings.md) for the training-side
evidence (especially the CVAE collapse), or jump to
[`04_root_cause_and_fixes.md`](04_root_cause_and_fixes.md) for what to do.
