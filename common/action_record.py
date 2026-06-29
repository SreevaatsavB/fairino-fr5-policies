"""
action_record.py — one shared schema for "actions over time", used by BOTH the
online robot rollout (common/deploy.py) and the offline test-set evaluation
(experiments/eval_on_test.py).

Keeping a single recorder means every saved rollout/eval has the *same* on-disk
layout, so tools/plot_actions.py can plot any of them without special-casing.

A record is an .npz of equal-length arrays (one row per timestep) plus a sidecar
.json holding metadata (policy, action_space, dimension labels). Ground truth is
optional: present for offline eval (we know the demonstrated action), absent for a
live robot rollout (there is none).

    rec = ActionRecorder(action_space="joint", policy="act", source="deploy")
    for t in range(T):
        rec.add(t=t, state=state_vec, pred=pred_action, executed=joints_cmd,
                gripper=gripper_cmd, gt=None, ik_failed=False)
    path = rec.save("policies/act/checkpoints/rollouts/deploy_2026.npz")
"""
import json
from pathlib import Path

import numpy as np


# Per-dimension labels by action space (6 motion dims + gripper = 7).
_DIM_LABELS = {
    "joint":     ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
    "delta_eef": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
}


def dim_labels(action_space: str, action_dim: int) -> list[str]:
    """Human-readable axis labels for an action vector; falls back to a#i if the
    space is unknown or the dim count does not match the known 7-D layout."""
    labels = _DIM_LABELS.get(action_space)
    if labels is not None and len(labels) == action_dim:
        return labels
    return [f"a{i}" for i in range(action_dim)]


class ActionRecorder:
    """Accumulate per-step action data and dump it to a .npz (+ .json sidecar)."""

    def __init__(self, action_space: str = "joint", policy: str = "",
                 source: str = "", extra_meta: dict | None = None):
        self.action_space = action_space
        self.policy       = policy
        self.source       = source            # "deploy" | "eval" | ...
        self.extra_meta   = dict(extra_meta or {})
        self._rows: list[dict] = []

    def add(self, t, state, pred, executed=None, gripper=None,
            gt=None, ik_failed=False):
        """Append one timestep. Arrays are copied to float32; None stays None."""
        def _vec(x):
            return None if x is None else np.asarray(x, dtype=np.float32).ravel()
        self._rows.append({
            "t":         float(t),
            "state":     _vec(state),
            "pred":      _vec(pred),
            "executed":  _vec(executed),
            "gt":        _vec(gt),
            "gripper":   (np.nan if gripper is None else float(gripper)),
            "ik_failed": bool(ik_failed),
        })

    def __len__(self):
        return len(self._rows)

    def _stack(self, key):
        """Stack a per-row field into (T, dim); rows with None become NaN."""
        vals = [r[key] for r in self._rows]
        present = [v for v in vals if v is not None]
        if not present:
            return None
        dim = len(present[0])
        out = np.full((len(vals), dim), np.nan, dtype=np.float32)
        for i, v in enumerate(vals):
            if v is not None:
                out[i] = v
        return out

    def save(self, path) -> Path:
        if not self._rows:
            raise ValueError("nothing to save — recorder is empty")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            "t":         np.array([r["t"] for r in self._rows], dtype=np.float32),
            "gripper":   np.array([r["gripper"] for r in self._rows], dtype=np.float32),
            "ik_failed": np.array([r["ik_failed"] for r in self._rows], dtype=bool),
        }
        for key in ("state", "pred", "executed", "gt"):
            arr = self._stack(key)
            if arr is not None:
                arrays[key] = arr

        np.savez(path, **arrays)

        action_dim = arrays["pred"].shape[1] if "pred" in arrays else 0
        meta = {
            "policy":       self.policy,
            "source":       self.source,
            "action_space": self.action_space,
            "action_dim":   action_dim,
            "dim_labels":   dim_labels(self.action_space, action_dim),
            "has_gt":       "gt" in arrays,
            "n_steps":      len(self._rows),
            **self.extra_meta,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return path


def load_record(path):
    """Load a record saved by ActionRecorder.save -> (arrays_dict, meta_dict)."""
    path = Path(path)
    npz = np.load(path, allow_pickle=False)
    arrays = {k: npz[k] for k in npz.files}
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return arrays, meta
