# Octo on the FR5

[Octo](https://octo-models.github.io/) is the generalist robot policy from the Octo team
(RAIL, Berkeley): a transformer with a diffusion action head, pretrained on **800k**
Open-X-Embodiment trajectories. This directory runs the **Octo team's pretrained weights**
on the FR5 in three workflows:

1. **`inference_pretrained.py`** — zero-shot inference (no finetuning)
2. **`finetune.py`** — finetune the pretrained model on the FR5 dataset
3. **`inference_finetuned.py`** — inference with the finetuned checkpoint

> **Octo is JAX/Flax, not PyTorch.** It does **not** use `common/train.py` or
> `common/deploy.py` (those are PyTorch/lerobot). It runs in its **own isolated venv**
> (`.venv-octo`) so the two dependency stacks never collide.

---

## 1. Setup (one time)

Octo + JAX 0.4.20 + TF 2.15 is a brittle dependency set; `setup_env.sh` pins the exact
combination verified to work, in an isolated Python 3.10 venv.

```bash
# Linux + NVIDIA GPU (CUDA 12) — recommended for finetuning:
bash policies/octo/setup_env.sh

# CPU-only (any platform; fine for inference, slow for finetuning):
bash policies/octo/setup_env.sh cpu

source .venv-octo/bin/activate
```

This installs the `octo` package, JAX, and the pinned `flax/optax/distrax/orbax/
tensorstore/tensorflow/transformers` stack, plus the lean extras the FR5 data adapter
needs (`pyarrow/opencv/pandas/pillow`). The Octo team weights download automatically on
first use from HuggingFace (`rail-berkeley/octo-small-1.5`, the default; or the larger
`octo-base-1.5`, ~547 MB).

---

## 2. Workflow 1 — zero-shot inference (no finetuning)

```bash
python policies/octo/inference_pretrained.py                    # random frame
python policies/octo/inference_pretrained.py --image frame.jpg  # a real wrist frame
python policies/octo/inference_pretrained.py --model hf://rail-berkeley/octo-base-1.5
```

Loads the pretrained model and runs it on an FR5 wrist image + language instruction.

> ⚠️ **Zero-shot actions are in Octo's pretrained action space** (normalized delta
> end-effector poses from its training mix), **not FR5 joint commands.** They will not
> drive the FR5 correctly until you finetune. This workflow proves the model loads and
> runs, and is the baseline for the finetuned comparison.

---

## 3. Workflow 2 — finetune on the FR5 dataset

```bash
python policies/octo/finetune.py --config policies/octo/config.yaml   # default: head_only
python policies/octo/finetune.py --mode full                          # train everything
python policies/octo/finetune.py --mode head_mlp_only
# quick smoke (tiny):
python policies/octo/finetune.py --root _smoke_dataset --steps 50 --batch-size 4 \
    --save-dir /tmp/octo_fr5
```

### Finetuning modes — Octo does NOT use LoRA

A common misconception. Octo's official recipe freezes parameters **by key pattern**
(`octo-models/octo` `scripts/configs/finetune_config.py`), not with LoRA adapters:

| Mode | What trains | `frozen_keys` | When |
|---|---|---|---|
| `head_only` *(default)* | heads only; transformer frozen | `octo_transformer.*` | **tiny datasets like the FR5** — leans on the 800k-trajectory visual prior, least overfitting |
| `head_mlp_only` | only the head MLP | `octo_transformer.*` + head map-head attn/probe | even more conservative |
| `full` | everything | `None` | Octo paper's default; needs more data + compute |

We use Octo's own `create_optimizer` (cosine LR, grad-clip 1.0, AdamW with no weight
decay on biases/norms) + `frozen_keys` — the exact pretrained recipe. Verified: in
`head_only`/`head_mlp_only` the transformer params don't move at all; in `full` they do.

What it does:

- FR5 action is **7-D (6 joints + gripper)** — this **matches** Octo's default action head
  (`action_dim=7, action_horizon=4`), so **no head surgery** is needed; the pretrained
  model is finetuned directly.
- FR5 **wrist camera → Octo `image_primary`** (256×256); the unused `image_wrist` slot is
  zeros and masked out.
- Actions are normalized with the dataset's own mean/std, saved as
  `checkpoints/fr5_action_stats.npz` for inference unnormalization.
- **Vision + language only, no proprioceptive state** — by design this sidesteps the
  proprioceptive shortcut (see [`docs/il_failure_modes.md`](../../docs/il_failure_modes.md)).

Checkpoints land in `policies/octo/checkpoints/<step>/` (orbax format).

---

## 4. Workflow 3 — inference with the finetuned checkpoint

```bash
python policies/octo/inference_finetuned.py \
    --checkpoint policies/octo/checkpoints --step 5000
```

Loads the finetuned weights + saved FR5 stats and produces actions in **FR5 space**
(6 joints + gripper, unnormalized). This is the model you'd deploy.

For the real robot, wrap it like `common/deploy.py`: at 30 Hz read the wrist frame, keep a
2-frame history window, call `sample_actions`, and send the first action of the returned
chunk (`action[:6]` joints, `action[6]` gripper). See
[`docs/inference.md`](../../docs/inference.md) for chunk-execution / temporal-ensembling.

---

## 5. Config (`config.yaml`)

| Key | Meaning |
|---|---|
| `model.pretrained` | Octo team weights — `octo-small-1.5` (27M, default) or `octo-base-1.5` (93M) |
| `model.window_size` | history frames (Octo default 2) |
| `model.action_horizon` | future actions predicted per step (Octo default 4) |
| `model.action_dim` | 7 = 6 joints + gripper (matches Octo's head) |
| `dataset.root` | LeRobot dataset path (same dataset as the torch policies) |
| `dataset.image_key` | which camera maps to `image_primary` |
| `finetune.*` | steps, batch size, lr, warmup, freeze, save dir |

---

## 6. Notes & gotchas

- **Head surgery:** only needed if your action dims differ from Octo's head
  (`7d × 4h`). The FR5 matches, so it's skipped. If you change `action_horizon`/`action_dim`,
  `finetune.py` will stop with a message — you'd then rebuild the head via
  `from_config` + `merge_params` (the standard Octo path).
- **Pins matter:** do not upgrade `numpy` (must be `<2`), `scipy` (`1.11.x`), `tensorflow`
  (`2.15`), or `jax` (`0.4.20`) inside `.venv-octo` — they form one fragile compatible set.
- **Verified:** all three workflows were run against `octo-small-1.5` (data adapter shapes,
  bound-module loss + gradient step, save→reload→FR5-space `sample_actions`). The Linux GPU
  box just needs `setup_env.sh` (CUDA) and the real dataset path.
