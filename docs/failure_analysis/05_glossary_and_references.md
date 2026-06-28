# 5. Glossary and References

Plain-language definitions of every term used in this folder, then all the papers with links.

---

## 5.1 Glossary

**ACT (Action Chunking Transformer)** — the policy here. A transformer that, from the current
observation, predicts a *chunk* of many future actions at once (here 100 steps), wrapped in a
CVAE to handle demonstration variability. Paper: Zhao et al. 2023.

**Action chunk** — a block of consecutive future actions predicted in one shot (vs. one action
per step). Reduces the effective decision horizon, which helps with compounding error.

**Behavioral cloning (BC)** — training a policy by supervised learning on (observation, action)
pairs from demonstrations. "Copy what the human did." No goal, no feedback, no causal reasoning.

**Causal confusion** — BC learns *correlations* between observations and actions, not *causes*.
It can latch onto an input that merely correlates with the right action (joint state, previous
action) and ignore the input that actually determines it (the camera). Paper: de Haan et al. 2019.

**Compounding error** — small per-step mistakes push the robot into states the expert never
visited, where it makes bigger mistakes, which compound. Under BC the total error grows like
O(T²) in the task length T. Paper: Ross et al. 2011 (DAgger).

**Conditional mean** — the average action for a given observation, `E[a|o]`. The mathematical
answer that mean-squared-error regression converges to — which is why plain BC mode-averages.

**Copycat / inertia problem** — the policy learns the shortcut "output ≈ my previous action,"
because consecutive expert actions are nearly identical. Near a slow-down it self-perpetuates
"keep (almost) stopping," producing a freeze. Paper: Wen et al. 2020.

**Covariate shift** — the train-time state distribution (expert states) differs from the
deploy-time state distribution (states the policy's own errors lead it into). The core reason
BC fails online while looking fine offline.

**CVAE (Conditional Variational Autoencoder)** — ACT's mechanism for multimodality. A latent
variable `z` encodes "which demonstration style/mode" so the decoder can commit to *one* mode
instead of averaging. Controlled by a **KL** term during training.

**DINOv2** — a large self-supervised Vision Transformer (from Meta). Used here as a **frozen**
image feature extractor in place of ACT's trainable ResNet18. Strong general visual features
without training them on your small dataset.

**EEF / TCP** — End-Effector / Tool Center Point: the position+orientation of the gripper in
3-D space. "EEF displacement" = how far the tool tip moved.

**Idle-action problem** — demonstrations contain "stay still" frames (operator pauses, especially
near the grasp). BC over-learns these as a confident "do nothing" rule, so the robot freezes
there. Paper: Chi et al. 2023 (Diffusion Policy) names it.

**KL term / KL divergence** — in the CVAE loss, the term that keeps the latent `z` informative
and close to its prior. If it collapses to ~0, the latent is unused → **posterior collapse**.

**Mode / multimodal** — a "mode" is one valid way to do the task (reach left vs. right; marker
here vs. there). Demonstrations are "multimodal" when several modes appear. BC averages them.

**Mode averaging** — predicting the average of multiple valid actions, which often lands on an
invalid in-between action (reach to the middle of two grasp points → grasp nothing).

**Posterior collapse** — a VAE/CVAE failure where the latent is ignored (KL → 0). Here it turned
ACT into a plain regressor → mode averaging. Classic with too-high KL weight. Origin of the fix
(KL annealing): Bowman et al. 2016.

**Proprioception / proprioceptive state** — the robot's sense of its own body: joint angles,
gripper opening, end-effector pose. The 7-D state input here (6 joints + gripper).

**Proprioceptive shortcut** — a specific causal confusion: the policy predicts the action from
the (clean, easy) joint state and ignores the (hard) cameras, so it reaches the same average
spot regardless of where the object is. Paper: Zhao et al. 2025.

**Proprioception dropout** — randomly zeroing the joint-state input during training so the
policy can't rely on it and is forced to use vision. You used rate 0.5. (`common/proprio.py`.)

**Temporal ensembling** — ACT's inference smoothing: query the policy every step and average the
overlapping predicted chunks with exponentially decaying weights. Here it *helps* (turning it off
made motion worse); it is **not** the cause of the freeze.

**Validation loss (val_l1)** — the prediction error on held-out expert frames. Here it is
**anti-correlated** with deploy success (the worst-val model was the deployed one), because a
lower val loss can just mean "better at predicting the object-blind mean."

---

## 5.2 References (all papers cited across this folder)

### The core failure modes
- **Causal confusion / proprioceptive shortcut**
  - de Haan, Jayaraman, Levine — *Causal Confusion in Imitation Learning* — NeurIPS 2019 —
    [arXiv:1905.11979](https://arxiv.org/abs/1905.11979)
  - Zhao et al. — *Do You Need Proprioceptive States in Visuomotor Policies?* — 2025 —
    [arXiv:2509.18644](https://arxiv.org/abs/2509.18644) · [project](https://statefreepolicy.github.io/)
- **Idle-action problem & mode averaging**
  - Chi et al. — *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* — RSS 2023 —
    [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- **Copycat / inertia**
  - Wen et al. — *Fighting Copycat Agents in Behavioral Cloning from Observation Histories* —
    NeurIPS 2020 — [arXiv:2010.14876](https://arxiv.org/abs/2010.14876)
  - Chuang et al. — *Resolving Copycat Problems in Visual Imitation Learning via Residual Action
    Prediction* — ECCV 2022 — [arXiv:2207.09705](https://arxiv.org/abs/2207.09705)
- **Covariate shift / compounding error**
  - Ross, Gordon, Bagnell — *A Reduction of Imitation Learning … to No-Regret Online Learning*
    (DAgger) — AISTATS 2011 — [PDF](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
  - Laskey et al. — *DART: Noise Injection for Robust Imitation Learning* — CoRL 2017 —
    [arXiv:1703.09327](https://arxiv.org/abs/1703.09327)
- **Gripper-state ambiguity / premature grasp**
  - *Disambiguate Gripper State in Grasp-Based Tasks: Pseudo-Tactile as Feedback Enables Pure
    Simulation Learning* — 2025 — [arXiv:2503.23835](https://arxiv.org/abs/2503.23835)

### The model and its mechanisms
- **ACT (the policy)**
  - Zhao, Kumar, Levine, Finn — *Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware* — RSS 2023 — [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- **CVAE posterior collapse / KL annealing**
  - Bowman et al. — *Generating Sentences from a Continuous Space* — CoNLL 2016 —
    [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
- **Multimodality alternatives**
  - Florence et al. — *Implicit Behavioral Cloning* — CoRL 2021 —
    [arXiv:2109.00137](https://arxiv.org/abs/2109.00137)
  - Shafiullah et al. — *Behavior Transformers: Cloning k Modes with One Stone* — NeurIPS 2022 —
    [arXiv:2206.11251](https://arxiv.org/abs/2206.11251)

### Related repo docs (this repository)
- [`docs/act.md`](../act.md) — ACT architecture from scratch (CVAE, z=0, temporal ensembling).
- [`docs/il_failure_modes.md`](../il_failure_modes.md) — the broader catalogue of BC failure modes.
- [`docs/proprioception_modes.md`](../proprioception_modes.md) — the full/dropout/none state switch.
- [`docs/action_spaces.md`](../action_spaces.md) — joint vs delta-EEF actions + the IK deploy path.
- [`docs/octo*.md`](../octo.md) — the state-free, pretrained alternative policy.

---

## 5.3 A note on rigor

Every claim in this folder is tied either to a **measured number from your logs**
(`experiment_data/…`, cited inline) or to a **named, real paper** (listed above). Where a fix's
payoff is an estimate rather than a measurement, it is labelled "rough/expected." The single most
important *measured* fact is your own ablation — **proprio influenced the action 2.2× more than
vision** — and the single most important *training* fact is the **CVAE posterior collapse** (KL →
~0 in every run). Those two, together, explain the object-blind average behaviour; the idle-action
freeze, the mid-air grasp, and the never-reaching-the-bowl follow from there.
