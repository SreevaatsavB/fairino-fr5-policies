"""
Fixes the action-scale problem behind the wandering pi0 policy. See README.md.

Two independent knobs, both aimed at the same thing — making the number the model
has to predict big enough to survive normalisation:

  delta=True        action = OFFSET from the chunk-start state, not the absolute
                    joint angle. Removes the part proprioception already gives the
                    model for free.                                   5.0x measured

  action_stride=K   the chunk samples every Kth frame instead of every frame, so
                    one step is K frames of motion instead of one.    5.0x at K=5

Measured on experiment_data/training_episodes_processed (chunk 50):

    per-step demo motion (signal)   0.0585 deg
    ABS   action std (normaliser)  13.099 deg  ->  signal/std = 0.0045
    DELTA action std (normaliser)   2.606 deg  ->  signal/std = 0.0225

The gripper (dim 6) is a 0/1 command, not a position, so it is never delta'd —
same split openpi uses for Libero (make_bool_mask(6, -1)).

IMPORTANT: action_stride > 1 changes the TIME between chunk entries. Training
sees entries K frames apart, so deploy must hold each one for K control steps or
the arm replays the trajectory K times too fast. hold() does that; run.py wires
it up automatically from the value recorded in the checkpoint.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from dataset import FR5Dataset  # noqa: E402

JOINTS = slice(0, 6)   # dims 0..5 are joint angles; dim 6 is the gripper command

_STAT_BLOCK = 4096     # ponytail: blocked accumulation — expanding every chunk at
                       # once is n_samples x chunk_size x 7 (~6 GB at 400 episodes)


def encode_action_space(delta: bool, stride: int) -> str:
    """'joint' | 'delta_joint' | 'delta_joint@5' — deploy.py's else-branch treats
    every one of these as joint-space, which is correct once the wrappers run."""
    base = "delta_joint" if delta else "joint"
    return f"{base}@{stride}" if stride > 1 else base


def parse_action_space(action_space: str):
    """'delta_joint@5' -> ('delta_joint', 5). Unknown/plain values -> stride 1."""
    base, _, stride = str(action_space).partition("@")
    return base, int(stride) if stride.isdigit() else 1


class DeltaJointDataset(FR5Dataset):
    """FR5Dataset with joint offsets and/or a strided action chunk.

    delta=False, action_stride=1 is exactly the stock dataset.
    """

    def __init__(self, *args, action_stride=1, delta=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_stride = max(1, int(action_stride))
        self.delta = bool(delta)
        # train.py copies ONLY info["action_space"] into the checkpoint, so the
        # stride rides along in that string ("delta_joint@5"). Keeps common/ +
        # policies/ untouched and makes the checkpoint self-describing — deploy
        # cannot silently run it at the wrong speed or without the state added.
        self.info = {**self.info,
                     "action_space":  encode_action_space(self.delta, self.action_stride),
                     "action_stride": self.action_stride}

    # ── chunk construction ────────────────────────────────────────────────────

    def _chunk_indices(self, frame_abs, ep_to):
        """Absolute row indices for the chunk, plus the padding mask.

        Clamping past the episode end reproduces the stock dataset's
        pad-with-last-action, and keeps get_stats able to mirror it exactly.
        """
        idx = frame_abs + np.arange(self.chunk_size) * self.action_stride
        is_pad = idx >= ep_to
        return np.minimum(idx, ep_to - 1), is_pad

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        if self.action_stride > 1:
            _, frame_abs, _, ep_to = self._samples[idx]
            rows, is_pad = self._chunk_indices(frame_abs, ep_to)
            sample["action"] = torch.tensor(
                np.array(self.df["action"].iloc[rows].tolist()), dtype=torch.float32)
            sample["action_is_pad"] = torch.from_numpy(is_pad)

        if self.delta:
            sample["action"][:, JOINTS] -= sample["observation.state"][JOINTS]
        return sample

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self):
        """State stats unchanged; action stats recomputed on what __getitem__
        actually emits. Reusing the absolute-action stats would defeat the whole
        change — the normaliser would still divide by the ~13 deg spread."""
        stats = super().get_stats()
        if not self.delta and self.action_stride == 1:
            return stats

        act = np.asarray(self.df["action"].tolist(), dtype=np.float64)
        sta = np.asarray(self.df["observation.state"].tolist(), dtype=np.float64)
        starts = np.fromiter((s[1] for s in self._samples), int, len(self._samples))
        ends   = np.fromiter((s[3] for s in self._samples), int, len(self._samples))
        k = np.arange(self.chunk_size) * self.action_stride

        dim = act.shape[1]
        n, total, sqsum = 0, np.zeros(dim), np.zeros(dim)
        lo, hi = np.full(dim, np.inf), np.full(dim, -np.inf)

        for i in range(0, len(starts), _STAT_BLOCK):
            s, e = starts[i:i + _STAT_BLOCK], ends[i:i + _STAT_BLOCK]
            rows = np.minimum(s[:, None] + k[None, :], e[:, None] - 1)
            d = act[rows]                                   # (B, chunk, dim)
            if self.delta:
                d[:, :, JOINTS] -= sta[s][:, None, JOINTS]
            flat = d.reshape(-1, dim)
            n     += len(flat)
            total += flat.sum(0)
            sqsum += (flat * flat).sum(0)
            lo = np.minimum(lo, flat.min(0))
            hi = np.maximum(hi, flat.max(0))

        mean = total / n
        var  = np.maximum(sqsum / n - mean * mean, 0.0)     # fp guard, not a fudge
        stats["action_mean"] = mean.astype(np.float32)
        stats["action_std"]  = np.sqrt(var).clip(1e-6).astype(np.float32)
        stats["action_min"]  = lo.astype(np.float32)
        stats["action_max"]  = hi.astype(np.float32)
        return stats


# ── inference wrappers (apply in this order: absolutise, then hold) ───────────

def absolutise(model):
    """Undo delta=True: model emits offsets, this adds the observed state back.

    The state is already an argument to predict(), so every downstream consumer
    (deploy's servo_j, eval's MAE, the action recorder) sees exactly what an
    action_space='joint' checkpoint produces. Returns the model.

    Idempotent: applying it twice would add the state twice — a silent ~13 deg
    offset on every joint command. eval wraps per episode, so this matters.
    """
    if getattr(model, "_delta_absolutised", False):
        return model
    original = model.predict

    def predict(obs_state, obs_image=None, task=None):
        action = original(obs_state, obs_image, task=task)
        offset = obs_state[..., JOINTS]
        while offset.dim() > action.dim():      # (1,7) obs vs (7,) action
            offset = offset.squeeze(0)
        out = action.clone()
        out[..., JOINTS] = action[..., JOINTS] + offset
        return out

    model.predict = predict
    model._delta_absolutised = True
    return model


def hold(model, stride):
    """Undo action_stride > 1: repeat each predicted action for `stride` control
    steps, because training spaced the chunk entries `stride` frames apart.

    Without this the arm executes the whole trajectory `stride` times too fast —
    at stride 5 that is 5x the demonstrated joint speed. Wrap AFTER absolutise so
    the cached action is already absolute (re-adding a moving state every step
    would drift). Idempotent. Returns the model.
    """
    if stride <= 1 or getattr(model, "_delta_held", False):
        return model
    original, reset = model.predict, model.reset
    state = {"n": 0, "action": None}

    def predict(obs_state, obs_image=None, task=None):
        if state["n"] % stride == 0:
            state["action"] = original(obs_state, obs_image, task=task)
        state["n"] += 1
        # clone: the cached tensor is handed out `stride` times, and a caller that
        # edits one copy in place must not corrupt the others
        return state["action"].clone()

    def reset_both():
        state["n"], state["action"] = 0, None
        reset()

    model.predict, model.reset = predict, reset_both
    model._delta_held = True
    return model


def for_inference(model, action_space):
    """Apply whatever the checkpoint's action_space string asks for, in order.

    Idempotent — eval calls this once per episode on the same model.
    """
    if getattr(model, "_delta_configured", False):
        return model
    base, stride = parse_action_space(action_space)
    if base == "delta_joint":
        absolutise(model)
    hold(model, stride)
    model._delta_configured = True
    print(f"[delta] inference: action_space={action_space!r} -> "
          f"{'offsets+state' if base == 'delta_joint' else 'absolute'}, "
          f"each action held {stride} control step(s)")
    return model
