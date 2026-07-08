"""
Shared helpers for the FR5 Octo scripts (JAX/Flax — runs in .venv-octo).

Octo is the generalist robot policy from the Octo team (RAIL, Berkeley), a
transformer with a diffusion action head trained on 800k Open-X trajectories.
It is JAX/Flax, so it lives OUTSIDE the repo's PyTorch/lerobot pipeline:
its own venv (.venv-octo), its own scripts, the Octo team's pretrained weights.

Verified against octo @ commit 241fb35 (octo-{small,base}-1.5):
  * window_size = 2, action_horizon = 4, action_dim = 7  (matches FR5: 6 joints + gripper)
  * obs the model reads: image_primary (B,W,256,256,3) uint8 + timestep_pad_mask (B,W) bool
  * sample_actions(obs, task, unnormalization_statistics=..., rng=...) -> (B, horizon, dim)
  * action stats are mean/std (NormalizationType.NORMAL)
"""

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pillow fallback
    cv2 = None

DEFAULT_MODEL = "hf://rail-berkeley/octo-small-1.5"  # Octo team weights (27M)
PRIMARY_IMAGE_SIZE = 256                              # Octo image_primary expected H=W
FR5_INSTRUCTION = "pick up the block and place it in the bin"
ACTION_DIM = 7                                        # 6 joints (deg) + gripper_norm [0,1]


# ── images ────────────────────────────────────────────────────────────────────

def resize_primary(rgb: np.ndarray) -> np.ndarray:
    """uint8 (H,W,3) RGB -> uint8 (256,256,3) for Octo's image_primary slot."""
    if cv2 is not None:
        return cv2.resize(rgb, (PRIMARY_IMAGE_SIZE, PRIMARY_IMAGE_SIZE),
                          interpolation=cv2.INTER_AREA).astype(np.uint8)
    from PIL import Image
    return np.asarray(
        Image.fromarray(rgb).resize((PRIMARY_IMAGE_SIZE, PRIMARY_IMAGE_SIZE))
    ).astype(np.uint8)


def read_image_rgb(path: str) -> np.ndarray:
    """Read an image file to uint8 (H,W,3) RGB."""
    if cv2 is not None:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


# ── model loading ───────────────────────────────────────────────────────────--

def load_octo(model_name: str = DEFAULT_MODEL, step: int | None = None):
    """Load a pretrained or finetuned Octo model.

    model_name: "hf://rail-berkeley/octo-base-1.5" (or -small), OR a local
                checkpoint directory written by finetune.py.
    """
    from octo.model.octo_model import OctoModel
    print(f"[octo] loading {model_name}" + (f" (step {step})" if step is not None else "") + " ...")
    model = OctoModel.load_pretrained(model_name, step=step)
    win = model.example_batch["observation"]["image_primary"].shape[1]
    horizon = model.config["model"]["heads"]["action"]["kwargs"]["action_horizon"]
    dim = model.config["model"]["heads"]["action"]["kwargs"]["action_dim"]
    print(f"[octo] loaded — window={win}  action_horizon={horizon}  action_dim={dim}")
    return model


def model_window_size(model) -> int:
    return int(model.example_batch["observation"]["image_primary"].shape[1])


# ── action statistics (Octo NormalizationType.NORMAL = mean/std) ────────────--

def action_stats(actions: np.ndarray) -> dict:
    """actions: (N, action_dim) -> Octo-style stats dict (mean/std/min/max/mask)."""
    a = np.asarray(actions, np.float32)
    return {
        "mean": a.mean(0),
        "std":  a.std(0).clip(1e-6),
        "min":  a.min(0),
        "max":  a.max(0),
        "mask": np.ones(a.shape[1], dtype=bool),   # unnormalize every dim
    }


def save_stats(path, stats: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_stats(path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def pick_pretrained_action_stats(model, dataset_key: str = "bridge_dataset") -> dict:
    """For zero-shot: borrow one source dataset's action stats so sample_actions can
    unnormalize. (Zero-shot actions are in Octo's pretrained action space, not FR5.)"""
    ds = model.dataset_statistics
    if dataset_key in ds:
        return ds[dataset_key]["action"]
    first = next(iter(ds.values()))
    return first["action"]
