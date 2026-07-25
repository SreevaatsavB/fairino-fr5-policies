# fairino-fr5-policies

Training and deploying imitation learning policies on the Fairino FR5 cobot. Data comes from teleoperation via an SO-101 leader arm, and the goal is to get the FR5 to autonomously replicate tasks like pick-and-place at 30 Hz.

This repo is set up as a centralised policy hub — one shared training/data pipeline in `common/`, and each policy type gets its own folder under `policies/`. So you can swap between ACT, Diffusion Policy, DiT etc. without touching any shared code.

---

## what's in here

| policy | folder | what it is | trains on |
|---|---|---|---|
| ACT | `policies/act/` | Action Chunking Transformer — CVAE + transformer, predicts 100-step chunks | CPU / local GPU |
| Diffusion Policy | `policies/diffusion/` | DDPM with a 1-D U-Net denoiser, 10-step DDIM at inference | local GPU |
| DiT + Flow Matching | `policies/dit_flow/` | Diffusion Transformer, flow-matching objective, swappable DINOv3/CLIP vision + language | local GPU |
| π0 | `policies/pi0/` | flow-matching VLA — 2B PaliGemma VLM + 300M action expert | **RunPod** (notebook) |
| π0.5 | `policies/pi05/` | π0 with a longer language context | **RunPod** (notebook) |
| π0-FAST | `policies/pi0_fast/` | π0's backbone, actions as discrete FAST tokens (no ODE) | **RunPod** (notebook) |
| Octo | `policies/octo/` | JAX generalist pretrained on 800k Open-X trajectories | isolated `.venv-octo` |

All the lerobot-family policies wrap **lerobot 0.5.1** and share the same FR5
episodes and `common/` pipeline. The small ones (ACT / Diffusion / DiT) train
locally with `common/train.py`; the π-family VLAs are too large for the robot box
and train on a rented GPU via the **self-contained RunPod notebooks** — see
[`docs/runpod_training.md`](docs/runpod_training.md). Octo lives in its own JAX
dependency stack (`.venv-octo`) — see [`docs/octo.md`](docs/octo.md).

**Deep dives:** every policy has a plain-English explainer under
[`docs/`](docs/README.md); runtime/latency is compared in
[`docs/inference.md`](docs/inference.md); quantization (QLoRA + inference) in
[`docs/quantization.md`](docs/quantization.md).

---

## setup

lerobot pins torch/torchvision below the newest releases so you need a venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

run everything from the repo root.

---

## getting data ready

raw recordings live in `episodes/episode_*/` — a ~100 Hz CSV log, a ~30 Hz wrist cam video, and a meta.json per episode. the converter resamples the CSV down to the camera rate (one row per frame) and outputs a LeRobot v3.0 dataset:

```bash
python common/convert_episodes.py --episodes episodes --out lerobot_dataset --extract-frames
```

`--extract-frames` pre-extracts JPEGs so training I/O is fast instead of seeking the video every step.

then update `dataset.root` in whatever config you're using to point at the output folder.

---

## training the small policies (ACT / Diffusion / DiT), locally

These train on the robot box or any local GPU (ACT even on CPU) via the shared loop.
The VLAs (π0 family) are separate — see the RunPod section below.

```bash
# ACT
python common/train.py --policy act --config policies/act/config.local.yaml

# Diffusion Policy
python common/train.py --policy diffusion --config policies/diffusion/config.local.yaml

# DiT + flow matching
python common/train.py --policy dit_flow --config policies/dit_flow/config.local.yaml
```

checkpoints go to `policies/<policy>/checkpoints/`. `best.pt` updates whenever val L1 improves. each checkpoint stores which policy it came from, so deploy just reads that and loads the right model automatically.

there's also a quick smoke test that runs convert → dataset → train step → checkpoint → predict in a few seconds:

```bash
python common/smoke_test.py act
python common/smoke_test.py diffusion
python common/smoke_test.py dit_flow
```

---

## training the VLAs (π0 / π0.5 / π0-FAST) on RunPod

The π-family carries a 2–3 B PaliGemma VLM and won't fit the robot box, so they
train on a rented GPU via **self-contained notebooks** (no `git clone` needed on the
pod — the whole pipeline is inline, and the checkpoints stay compatible with
`deploy.py`):

```
notebooks/train_pi0_runpod.ipynb        notebooks/train_pi05_runpod.ipynb
notebooks/train_pi0_fast_runpod.ipynb   notebooks/convert_and_push_dataset.ipynb
```

Quick version: launch a **Python-3.12** GPU pod, set `HF_TOKEN` (with the PaliGemma
license accepted) and `HF_DATASET_REPO`, pick `QUANTIZE="nf4"` for a 24–48 GB card,
and run top-to-bottom. It pulls the dataset, finetunes (QLoRA on the VLM + full
action expert), evaluates every held-out episode, and pushes `best.pt` to the Hub.
Full guide — every knob, VRAM/speed tuning, resume, evaluation dashboard:
**[`docs/runpod_training.md`](docs/runpod_training.md)**.

Quantization (what NF4/QLoRA is, training vs inference, the Turing+ requirement):
**[`docs/quantization.md`](docs/quantization.md)**.

---

## deploying on the robot

From a local checkpoint:

```bash
python common/deploy.py --checkpoint policies/act/checkpoints/best.pt
```

Or pull a checkpoint the RunPod notebook pushed, straight from the Hub:

```bash
python common/deploy.py --hf-repo <you>/fr5-pi0-lora                 # downloads best.pt
python common/deploy.py --hf-repo <you>/fr5-pi0-lora --quantize nf4  # NF4 for a low-VRAM GPU
```

Runs at 30 Hz for 150 steps by default (`--steps N` to change). `--no-image` falls
back to state-only if the camera isn't connected. `--quantize nf4|int8`
post-quantizes a π-family checkpoint for a small GPU even if it was trained in bf16
(Turing+ required — see [`docs/quantization.md`](docs/quantization.md)). Inference
smoothing knobs `--te-coeff` / `--n-action-steps` work on existing checkpoints with
no retraining ([`docs/inference.md`](docs/inference.md)). Every rollout logs its
predicted actions to `<ckpt_dir>/rollouts/` ([`docs/evaluation.md`](docs/evaluation.md)).

---

## adding a new policy

drop a folder under `policies/<name>/` with:
- `model.py` — implements `build_model(cfg, stats, device)` returning a model with `forward(obs, actions, pad, image, task) -> (loss, l1, kl)`, `reset()`, and `predict(obs, image, task)`
- `config.yaml`, `config.local.yaml`, `config.smoke.yaml`

that's it. `common/train.py --policy <name>` picks it up automatically.

---

## repo layout

```
common/
  convert_episodes.py    raw episodes → LeRobot dataset
  dataset.py             parquet + video → PyTorch DataLoader
  train.py               shared training loop (--policy <name>)
  deploy.py              load any checkpoint (local or --hf-repo), run on the FR5
  vla_pretrained.py      π-family: verified pretrained load, LoRA, NF4/int8 quantize
  lerobot_patches.py     transformers/torch compat shims for lerobot 0.5.1
  smoke_test.py          quick end-to-end sanity check
policies/
  act/  diffusion/  dit_flow/     small policies — train locally
  pi0/  pi05/  pi0_fast/          VLAs — train on RunPod (notebooks/)
  octo/                           JAX generalist (isolated .venv-octo)
notebooks/
  train_pi0_runpod.ipynb         self-contained RunPod training + eval + Hub push
  train_pi05_runpod.ipynb
  train_pi0_fast_runpod.ipynb
  convert_and_push_dataset_v2.ipynb  400-ep multi-task set → LeRobot → Hub (videos-only)
  convert_and_push_dataset.ipynb     v1 (134-ep single-task), kept for reproducibility
docs/
  runpod_training.md     the VLA notebooks, every knob, VRAM/speed tuning
  quantization.md        QLoRA training + post-training inference quantization
  inference.md           runtime/latency across all policies
  evaluation.md          action logging + GT-vs-prediction
eda/
  eda.ipynb              explore the dataset before training
```

---

## hardware

- **Robot**: Fairino FR5 (6-DOF cobot)
- **Leader arm**: SO-101 for teleoperation
- **Camera**: Intel RealSense D405 (wrist-mounted), 640×480 → resized to 224×224
- **Control rate**: 30 Hz
- **Action space**: 6 joint angles (deg) + gripper (0–1 normalised)
- **State space**: 6 actual joint positions (deg)
