# Reading list — human/UMI data for VLAs, and world models → world action models

Compiled 2026-08-26 from the primary sources linked below. Every arXiv link was
verified against the arXiv API at compile time; the few papers cited by title
only could not be verified that day (search by title).

## Framing

Two threads, converging in 2026:

- **A — action-labelled data without robots.** UMI gives you *actions* (EE pose +
  gripper) from a human holding a gripper; human-hand video gives you *no* actions.
  The literature is about closing four gaps: embodiment, viewpoint, kinematic
  feasibility, action-space alignment.
- **B — world models → world action models.** A world model predicts the future; a
  WAM couples the prediction to action output. The July 2026 tutorial
  ([2607.00836](https://arxiv.org/html/2607.00836v1)) defines a WM as *"a model to
  predict how its future observation o_{t+1} or state x_{t+1} evolves under action
  a_t"* and a WAM as a policy that *"extends world models by explicitly associating
  predicted future observations or states with actions."* NVIDIA's
  [framing](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/):
  *"a policy that starts from a pretrained world-model or video backbone and adapts
  it to … emit corresponding actions."*

They meet in XR-1 (100k h UMI pretraining) and
[τ₀-WM](https://arxiv.org/abs/2606.01027) (~27k h of *"real-robot teleoperation,
UMI-style interaction, egocentric human videos"* in one world model).

---

## Thread A — human/UMI data → VLAs (read in this order)

### 1. The device and what to collect
| paper | why |
|---|---|
| [UMI](https://arxiv.org/abs/2402.10329) (RSS 2024) | the mechanism: hand-held gripper, wrist fisheye, SLAM for EE pose. *Why* the action is EE-pose-in-gripper-frame: it is the only thing a human and a robot share |
| [Data Scaling Laws in Imitation Learning](https://arxiv.org/abs/2410.18647) (ICLR 2025, UMI-based) | generalisation scales with **environments × objects**, not demos per task; ~32 env-object pairs saturate. Decides how collection time is spent |

### 2. UMI at pretraining scale — compare side by side
| paper | claim |
|---|---|
| [XR-1](https://arxiv.org/abs/2607.15330) (2026) | 100k h UMI, auto-captioned 2-s segments, unified EE-frame orientation across embodiments, 10k h post-training. Analysis in `xr1_eef/README.md` |
| [RDT2](https://arxiv.org/abs/2602.03310) (Feb 2026) | 10k h UMI, 7B VLM, RVQ + flow matching + distillation; *"zero-shot"* transfer to *"even robotic platforms"* never seen. The strongest form of the embodiment-free thesis |
| [VISTA](https://arxiv.org/abs/2606.04708) (Jun 2026) | the honest paper on **why UMI data is hard for VLAs**: fisheye wrist views are OOD for pretrained VLMs; human trajectories *"violate kinematic limits, incur collisions, or exceed controller bandwidth."* Fixes: a wrist-fisheye VQA set to re-ground the VLM, and a physical-validation pipeline whose scores are *"strongly predictive of deployment success"* |

### 3. Cross-embodiment transfer of UMI policies (skim)
[UMI-on-Legs](https://arxiv.org/abs/2407.10353), [UMI-on-Air](https://arxiv.org/abs/2510.02614)
— policy stays embodiment-agnostic, the *controller* absorbs the embodiment.
[DexUMI](https://arxiv.org/abs/2505.21864) (hands). [FastUMI](https://arxiv.org/abs/2409.19499)
+ [FastUMI-100K](https://arxiv.org/abs/2510.08022) (cheaper hardware, public dataset).

### 4. Human hands with no gripper — the harder gap
[EgoDex](https://arxiv.org/abs/2505.11709) (829 h egocentric + 3D hand pose),
[Being-H0](https://arxiv.org/abs/2507.15597) (VLA pretraining from human video via
hand-motion tokens), [EgoVLA](https://arxiv.org/abs/2507.12440) (predict wrist/hand
motion, retarget), [Spatial-Aware VLA Pretraining from Human Videos](https://arxiv.org/abs/2512.13080).
By title (IDs unverified): *Humanoid Policy ~ Human Policy* (2025), *Phantom:
Training Robots Without Robots Using Only Human Videos* (2025), *Motion Tracks*
(2025, point tracks as the shared action space).

### 5. No actions at all — latent actions from video
[LAPA](https://arxiv.org/abs/2410.11758) (VQ-VAE latent actions between frames; small
robot set maps latent→real) → [UniVLA](https://arxiv.org/abs/2505.06111) (DINO
feature space, language strips task-irrelevant motion) → [villa-X](https://arxiv.org/abs/2507.23682),
[Motion-Focused Latent Action](https://arxiv.org/html/2606.18955) (2026).
[GR00T N1](https://arxiv.org/abs/2503.14734) as the synthesis: the data pyramid
web video → human video → synthetic neural trajectories → robot data.

**Take-away from A:** every paper picks an *interface* between human and robot —
EE pose in the tool frame (UMI/XR-1/RDT2), hand keypoints (EgoDex/Being-H0), point
tracks (Motion Tracks), learned latents (LAPA/UniVLA) — then pays for the residual
gap with a controller (UMI-on-Legs), a validation filter (VISTA), or a small robot
dataset (everyone).

---

## Thread B — world models → world action models

### Stage 0 — latent world models for control
[World Models](https://arxiv.org/abs/1803.10122) (2018) → [Dreamer](https://arxiv.org/abs/1912.01603)
→ [DreamerV3](https://arxiv.org/abs/2301.04104): learn latent dynamics, train the
policy inside imagination. [TD-MPC2](https://arxiv.org/abs/2310.16828) (plan in
latent space), [IRIS](https://arxiv.org/abs/2209.00588) (transformer WM),
DayDreamer (2022, real robots). Read Dreamer + DreamerV3 properly; skim the rest.

### Stage 1 — video generation *is* a world model
[UniPi](https://arxiv.org/abs/2302.00111) (2023) — **the prototype WAM**: generate a
video of the task, run inverse dynamics. Every "cascaded" WAM descends from it.
[UniSim](https://arxiv.org/abs/2310.06114), [GAIA-1](https://arxiv.org/abs/2309.17080),
[Genie](https://arxiv.org/abs/2402.15391) (action-conditioned from unlabelled video;
Genie 3 is a 2025 blog, no paper), [Cosmos](https://arxiv.org/abs/2501.03575).

### Stage 2 — latent (JEPA-style) world models for planning
[V-JEPA 2](https://arxiv.org/abs/2506.09985) (predict in embedding space; small
action-conditioned head plans zero-shot from image goals; 2.1 in 2026).
[DreamerV4 — Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527).
[FLARE](https://arxiv.org/abs/2505.15659) — the same idea *inside* a policy.

### Stage 3 — WAMs proper
Taxonomy per the [tutorial](https://arxiv.org/html/2607.00836v1) and
[Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM): **cascaded** vs **joint**;
the [May 2026 survey](https://arxiv.org/html/2605.00080v1) adds WM-as-simulator (RL,
evaluation) and WM-for-data-generation.

| family | papers |
|---|---|
| cascaded | UniPi → [Video Prediction Policy](https://arxiv.org/abs/2412.14803) (use the video model's *features*, don't decode) → [Gen2Act](https://arxiv.org/abs/2409.16283) |
| joint, autoregressive | [GR-1](https://arxiv.org/abs/2312.13139) → [WorldVLA](https://arxiv.org/abs/2506.21539) → [VLA-JEPA](https://arxiv.org/abs/2602.10098) |
| joint, diffusion | [Unified Video Action](https://arxiv.org/abs/2503.00200) → [UWM](https://arxiv.org/abs/2504.02792) → **[Cosmos Policy](https://arxiv.org/abs/2601.16163)** (fine-tune a video model for control — cleanest statement of the thesis) → [DreamZero](https://arxiv.org/abs/2602.15922), [LingBot-VA](https://arxiv.org/abs/2601.21998), [MotuBrain](https://arxiv.org/pdf/2604.27792) |
| WM for data | [DreamGen](https://arxiv.org/abs/2505.12705) — "neural trajectories": a video model dreams demos, a latent-action model labels them |
| WM as simulator / evaluator | [Ctrl-World](https://arxiv.org/abs/2510.10125), [Genie Envisioner](https://arxiv.org/abs/2508.05635) |
| WM on UMI + human video | [τ₀-WM](https://arxiv.org/abs/2606.01027) |

### Stage 4 — the 2026 pushback (read last)
[Fast-WAM](https://arxiv.org/abs/2603.16666) — *"do WAMs need test-time future
imagination?"*; [ImageWAM](https://arxiv.org/abs/2606.19531) — *"or just image
editing?"*; [Is the Future Compatible?](https://arxiv.org/pdf/2605.07514) — dynamic
consistency failures; [WEAVER](https://arxiv.org/html/2606.13672);
[RoboTrustBench](https://arxiv.org/pdf/2606.01600). Recurring finding: most of the
gain is the **pretrained video representation**, not imagining futures at inference.

---

## Suggested order (≈14 core papers)

UMI → Data Scaling Laws → XR-1 (§3 with the action formula in mind) → RDT2 → VISTA
→ LAPA → GR00T N1 → Dreamer → UniPi → V-JEPA 2 → Video Prediction Policy → Cosmos
Policy → DreamGen → Fast-WAM.

Then branch via [Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM) and
[awesome-UMI-Papers](https://github.com/huangjund/awesome-UMI-Papers); keep the
[survey](https://arxiv.org/html/2605.00080v1) open as a map.

## Why it matters for the FR5

Three papers change what we do next: **VISTA**'s validation pipeline is the missing
step for our human-paced demos before they feed a VLA; **RDT2 vs XR-1** is the live
argument over whether ~10k h of UMI alone gets zero-shot transfer or robot
post-training is still required (our 5 h says post-training); **Cosmos Policy /
DreamGen** are the WAM route to try once the XR-1 baseline is gated — DreamGen in
particular because it manufactures the one thing we are short of, demonstrations.
