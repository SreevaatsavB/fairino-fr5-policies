# Xiaomi-Robotics-1 (XR-1) on the FR5 — analysis and integration plan

Sources read for this, in order of authority: the released code
(`github.com/XiaomiRobotics/Xiaomi-Robotics-1`, commit on `main` 2026-08-03), the
technical report (arXiv 2607.15330), the model card, the project page, and one real
episode of their post-training demo dataset (`XiaomiRobotics/xr1_post_train_demo`).
Where the paper and the code disagree, the code wins — it is what trains and runs.

**Decision this folder implements:** the FR5 policy predicts **end-effector poses**
from now on, in XR-1's exact representation, so that XR-1-5B can be post-trained on
our data and deployed with their runtime.

---

## 1. What XR-1 is

| | |
|---|---|
| backbone | Qwen3-VL-4B-Instruct (4.4 B) — vision + language, **fully trained** during post-training (only the input embedding table is frozen, `XR1.py:227`) |
| action head | DiT, 36 layers × hidden 1024 (0.6 B), coupled to the VLM as a *Mixture-of-Transformers*: the DiT attends into the VLM's KV cache |
| action generation | flow matching, **5 Euler steps** (`num_steps = 5`), timesteps sampled `Beta(1.5, 1.0)` during training |
| auxiliary head | "Choice policy": the VLM itself regresses 5 candidate chunks + a score from `<a_0>…<a_29>` and `<score>` tokens; `L = L_flow + L_regression + 0.1·L_NTP` |
| async execution | with p = 0.5 a random 1–6-step **action prefix** (the actions already executing) is given to the DiT, so inference can overlap with motion |
| sizes released | **5B only** (VLM 4.4 B + DiT 0.6 B; `model_states.pt` 10.2 GB bf16). The 2B/10B variants in the paper are not on the Hub |
| pre-training | 100k h of UMI hand-held-gripper trajectories, auto-captioned by Qwen3.5-27B into 2-s "state transition" descriptions |
| post-training | ~10k h: 7.2k h in-house robots + 1k h instruction-labelled UMI + Bridge/RT-1/DROID |
| results | RoboCasa 74.5, RoboCasa365 57.4, VLABench 59.1; real-world new-task adaptation **75 % @ <10 h/task vs π0.5 40 %**, 85 % @ <40 h |

The claim that matters for us is the last row: the whole point of the model is
cheap adaptation to a new embodiment from a few hours of demos. We have 5.06 h.

---

## 2. The action representation — exactly, from `json_dataset.py:_arm_action`

```python
rotm = proprios.ee_rotm[t].reshape(3, 3)            # current EE orientation, base frame
pos  = proprios.ee_pos[t]                            # current EE position, base frame, METRES
for i in 0..29:
    Δpos_i  = rotm.T @ (actions.ee_pos[t+i] - pos)   # translation expressed in the CURRENT TOOL frame
    Δrot_i  = rotm2aa(rotm.T @ actions.ee_rotm[t+i]) # rotation to target, as axis-angle, in the tool frame
    Δgrip_i = actions.gripper[t+i] - proprios.gripper[t]
```

This is the paper's `a_{t+i} = (T^EE_Base,t)^{-1} · T̂^EE_Base,t+i` in code. Three
properties that matter:

1. **Every entry is relative to the state at chunk start** — the same idea as our
   `delta_joint/`, but in Cartesian tool-frame coordinates. Our whole diagnosis of
   the 07-28 failure ("the model was asked for the absolute position and the
   0.06° that matters vanished in normalisation") is exactly what this
   representation avoids. XR-1 was designed around it.
2. **The tool frame is the reference, not the base frame.** "Move 2 cm forward"
   is the same numbers for a UMI gripper, a Xiaomi arm and an FR5 — that is what
   makes the 100k h of UMI data transfer. It also means the *orientation of our
   TCP frame* must match theirs (§6, the one real risk).
3. **Normalisation is per chunk position**: `mean`/`std` have shape `(30, 60)`.
   Entry 0 is normalised by entry 0's own std. The horizon-inflation problem we
   measured in `delta_joint/README.md` (chunk 50 × stride 5 pooled the normaliser
   to 9.2° and buried entry 0) cannot happen here.

Packed layout (`io.py:ACTION_PARTS`), 60 dims, single-arm FR5 uses the left slot:

| dims | content | FR5 |
|---|---|---|
| 0–2 | left EE Δpos (tool frame, m) | **used** |
| 3–5 | left EE Δrot axis-angle (rad) | **used** |
| 6 | left gripper Δ | **used** |
| 8–14 | right arm | zeros |
| 16 | waist Δ | zero |
| 17–19 | base velocity | zeros |

**State** (`compose_state`, 60 dims, quantile-normalised to [−1, 1] with `q01/q99`):
`left_arm_joint` (≤7 joints, **radians**) at 0–6, `left_gripper` at 7, right arm at
8–15, rest zero. Note the state is *joints*, not EE pose — the EE pose is only used
to build the action deltas and to recover absolute targets at deploy.

**Horizon:** `action_length = 30` at the data's 30 fps = **1.0 s**. Their demo video
is 960×720 @ 30 fps; ours is 640×480 @ 30 fps — same rate, no resampling.

---

## 3. Data format (`docs/data_format.md`) and what a real episode contains

One JSON per episode + one mp4 per camera. Measured on `json1.json` of their demo:

```
num_frames 489            trajectory_type "ongoing"  (allowed: success | ongoing | invalid)
proprios.left_ee_pos      metres      range x 0.079–0.137  (per-frame |Δ| ≈ 0.2 mm)
proprios.left_ee_rotm     3×3 flattened, det 1.000
proprios.left_arm_joint   6 joints, RADIANS (−0.36 … 1.57)
proprios.left_gripper_pos ≈ −2.97   (their units; only the delta and the quantile range matter)
actions.left_ee_pos       ≈ proprio + 2.6 mm   → action[t] is the commanded target for frame t
instruction               "Load washer."
```

Augmentation (`_augment`): brightness ±32/255, contrast 0.5–1.5, saturation 0.5–1.5,
**hue 0.0**, each applied with p = 0.5 independently; images resized to ≤160 k px
(multiples of 32). Same philosophy we arrived at for the colour-conditioned task:
photometric only, no hue, no crop.

---

## 4. Post-training recipe (`configs/trainer/deepspeed.yaml`, `XR1.py`)

| | |
|---|---|
| trainable | all of Qwen3-VL except the token embedding; all of the DiT and projectors |
| optimiser | FusedAdam, betas (0.9, 0.95), **wd 0.1**, clip 1.0 |
| schedule | warmup 500 → cosine, **max_lr 2e-5, min_lr 5e-6** |
| steps / batch | 10 000 steps, batch 48 per process, `training_repeat = 4` (each sample gets 4 noise draws) |
| precision | bf16-mixed, flash-attn 2, gradient checkpointing on the vision tower and every FFN |
| data sampling | every frame of every episode is a chunk start; padded with last-action repeat at episode end |
| env | `torch 2.8.0`, `transformers == 4.57.1` exactly, `deepspeed 0.18.9`, `lightning 2.5.3`, `decord` |

**Compute reality for the 5B full fine-tune:** bf16 weights 10 GB + fp32 master
20 GB + Adam m/v 40 GB + grads ≈ 80 GB before activations. **It does not fit one
A100-80.** Their reference runs are multi-GPU. Options, cheapest first:

1. **Freeze the VLM, train the DiT + projectors (~0.6 B)** — fits an A100-80 with
   room to spare, and is the standard first move for a new embodiment. Two lines
   in `xr1.__init__` (`self.vlm.requires_grad_(False)`); the repo has no flag for it.
2. 2–4 × A100-80 with ZeRO-2/3 for the full recipe.

Inference: 10.2 GB bf16 + activations ≈ 13 GB → fits the robot PC's 16 GB card,
which is Blackwell so flash-attn 2 is fine.

---

## 5. FR5 → XR-1 mapping (what `fr5_to_xr1.py` does)

| XR-1 field | FR5 source | conversion |
|---|---|---|
| `left_ee_pos` | `observation.eef_pose[:3]` (mm) | ÷ 1000 → m |
| `left_ee_rotm` | `observation.eef_pose[3:6]` = `rx, ry, rz` (deg) | **extrinsic XYZ**: `R = Rz(rz)·Ry(ry)·Rx(rx)` = scipy `from_euler('xyz', …, degrees=True)`; wrap-safe (our data has 41 ±180° jumps in `rx`) |
| `left_arm_joint` | `observation.state[:6]` (deg) | → rad |
| `left_gripper_pos` | `observation.state[6]` (0 = open, **1 = closed**) | as-is |
| `actions.left_ee_*` | proprio of frame **t+1** (last frame repeats) | matches their demo, where `action[t] ≈ proprio[t] + one step` |
| `actions.left_gripper_pos` | `action[6]` (commanded gripper) | as-is |
| right arm, waist, base | — | zeros (state dims with `q01 == q99 == 0` are treated as padding by `validate_quantiles`) |
| ego view | `videos/observation.images.scene_cam/chunk-000/file-{ep:03d}.mp4` | fixed overview D435i |
| wrist-left view | `videos/observation.images.wrist_cam/…` | D405 on the wrist |
| wrist-right view | none | prompt carries **two** `<image>` placeholders; the loader accepts any count that matches `images` |
| instruction | `data.task_index → meta/tasks.parquet` | the 9 shared sentences from 2026-07-30 (**not** `meta/episodes.task`, which still holds the old per-episode strings) |

### The Euler convention — verified, not assumed

Fairino's `robot_types.h`: *"rx: Rotation Angle about **fixed** axis X … ry … fixed
axis Y … rz … fixed axis Z, unit: deg"* → extrinsic XYZ. Confirmed on our own data
with a test that needs no robot: on the 235 frames where **only joint 6** moved, the
relative rotation expressed in the tool frame must be a pure roll about the tool's
z-axis —

```
extrinsic XYZ  (R = Rz·Ry·Rx)   |axis·z| = 1.000   angle/frame 0.42° = |ΔJ6| 0.42°   ✓
intrinsic XYZ  (R = Rx·Ry·Rz)   |axis·z| = 0.772                                       ✗
```

`test_fr5_to_xr1.py` re-runs this whenever the parquet is present.

### Signal scale, measured by the converter on all 400 episodes (546,949 chunk starts)

| chunk entry | Δpos std (m) | Δrot std (rad) |
|---|---|---|
| 0 (now) | 0.00075 / 0.00066 / 0.00072 ≈ **0.7 mm** | 0.0026 ≈ **0.15°** |
| 29 (1 s ahead) | 0.0188 / 0.0164 / 0.0182 ≈ **18 mm** | 0.058 ≈ **3.4°** |

A 25× spread between entry 0 and entry 29 — exactly the situation that broke our
pooled normaliser in `delta_joint/` (chunk 50 × stride 5). Because XR-1 normalises
**per chunk position**, entry 0 is divided by 0.7 mm and entry 29 by 18 mm; each
entry lands at unit scale. State quantiles (rad): J1 −2.37…−1.07, J2 −2.06…−1.07,
J3 1.75…2.69, J4 −3.60…−2.26, J5 −2.11…−1.06, J6 −0.98…0.80; gripper 0…1. The
whole stats pass takes 25 s on a laptop.

---

## 6. Risks, honestly ranked

1. **Tool-frame orientation alignment.** The paper: *"we unify the orientation of
   the end-effector frames across all robot data and UMI data"*. Which axis points
   out of the gripper in their canonical frame is **not documented**. If the FR5
   TCP frame differs by a fixed rotation, the pretrained prior maps "forward" to
   the wrong axis and post-training has to unlearn it. `fr5_to_xr1.py` exposes
   `TOOL_FRAME_FIX` (a constant 3×3 applied to every rotm) for this; determining
   it needs one look at their UMI/robot frames, or a short probe run. Default
   identity, flagged loudly in the output.
2. **Two views, not three.** The pretrained model has always seen three. The prompt
   format is legal with two, but it is a distribution shift the post-training has
   to absorb. Alternative via `--right-wrist ego|wrist` duplicates a view into the
   third slot — we know from 07-26 that feeding a wrong view into a slot can be
   worse than an empty one, so the default is two views.
3. **Binary gripper.** Theirs is a continuous position; ours is a 0/1 command. The
   delta is then ∈ {−1, 0, 1}; the quantile-normalised state is ±1. Works, but
   the model gets no "half-closed" signal.
4. **Compute** (§4): a single A100-80 forces the frozen-VLM variant.
5. **Deployment is untested on hardware.** `deploy_fr5_xr1.py` closes the loop
   (state → XR-1 → `recover_action` → Fairino RPY → IK → `servo_j`) and reuses the
   proven `fr5.py` primitives, but has not driven the arm.

What is *not* a risk: rates (both 30 Hz), units (converted and asserted), the
Euler convention (proven), and the instruction format (their `_prompt` appends
`/no_cot` itself).

---

## 7. Files

| file | what |
|---|---|
| `fr5_to_xr1.py` | LeRobot-v2 → XR-1 JSON episodes + the `(30,60)` mean/std and `(1,60)` q01/q99 config, computed with XR-1's own delta formulas |
| `deploy_fr5_xr1.py` | robot-PC loop against their inference server: 2-view client, `recover_action`, rotm → Fairino RPY, IK + `servo_j`, receding horizon |
| `test_fr5_to_xr1.py` | Euler convention (incl. the real-data J6 test), delta ↔ recover round-trip, wrap safety, converter schema/shape checks, stats layout |
| `xr1_stats.py` | normalisation statistics for **any** XR-1-format dataset — the single implementation of their per-position mean/std and state quantiles; the converter routes through it |
| `../notebooks/train_xr1_runpod.ipynb` | the RunPod notebook, **dataset-agnostic**: any XR-1-format dataset (HF repo or dir; LeRobot→XR-1 conversion opt-in), their pinned env + repo, five env-gated patches (`/dev/shm`, CSVLogger, workers, freeze-VLM, JSON cache), seeded held-out split, stats, one sample through **their** loader before torchrun, training with `metrics.csv` tailing, push in the deploy layout |

```bash
python xr1_eef/fr5_to_xr1.py /tmp/ds_v2_edit --out /workspace/xr1_data --dry-run   # stats + schema, no videos needed
python xr1_eef/fr5_to_xr1.py /tmp/ds_v2_edit --out /workspace/xr1_data              # writes json/ + configs/fr5.yaml
python xr1_eef/xr1_stats.py /workspace/xr1_data/json --out configs/data/fr5.yaml   # stats for ANY XR-1-format json dir
python xr1_eef/test_fr5_to_xr1.py
```

## 8. Next

1. Run the converter on the pod against the v2 dataset → `configs/fr5.yaml`.
2. Post-train with the VLM frozen first (fits one A100-80, ~1 h at 10k steps is
   optimistic; expect 3–5 h). Check `val` the way we now know to: gate criteria,
   not the loss.
3. Resolve risk 1 before robot time — it is the one thing that could make a
   correct pipeline steer the wrong way.
