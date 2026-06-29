#!/usr/bin/env python3
"""
eval_on_test.py — run a trained policy over the HELD-OUT test episodes and plot the
predicted action vs the ground-truth (demonstrated) action, one subplot per action
dimension, over time.

This is the offline counterpart to a robot rollout: instead of acting on the FR5, we
replay each test episode's recorded observations through the policy and compare what
it *would* command against what the human actually did. It is OPEN-LOOP / teacher-
forced — every observation comes from the dataset, not from the policy's own previous
action — so it measures single-step prediction accuracy, not closed-loop drift. (It is
the right tool for "does the policy reproduce the demonstrated actions?"; closed-loop
behaviour still has to be checked on the robot.)

Cross-policy: uses the shared model.predict() + model.reset() that every lerobot-style
policy exposes (act, diffusion, dit_flow, pi0, pi05, pi0_fast), so it works for all of
them. The test split is FR5Dataset.episode_split(val_frac, seed) — the same held-out
episodes train.py never trained on. (Octo has its own inference pipeline and is not
covered here.)

Usage:
    python experiments/eval_on_test.py --ckpt policies/act/checkpoints/best.pt
    python experiments/eval_on_test.py --ckpt best.pt --episodes 3 --max-steps 300
    python experiments/eval_on_test.py --ckpt best.pt --episodes 0 5 9   # specific eps

Writes <ckpt_dir>/eval/ep<NNN>.npz (+ .json + .png) per episode and prints a per-
dimension MAE table.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "tools"))
from dataset import FR5Dataset          # noqa: E402
from action_record import ActionRecorder, load_record  # noqa: E402
from plot_actions import plot_record    # noqa: E402


def _load_policy_module(policy):
    spec = importlib.util.spec_from_file_location(
        f"policy_{policy}", REPO / "policies" / policy / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_model_images(sample_imgs, device):
    """Mirror train.py/deploy.py: one camera -> a single (1,C,H,W) tensor (what the
    single-cam policies expect); several cameras -> a {key: tensor} dict (only ACT
    consumes the dict form). None when the policy is state-only."""
    if not sample_imgs:
        return None
    imgs = {k: v.unsqueeze(0).to(device) for k, v in sample_imgs.items()}
    return next(iter(imgs.values())) if len(imgs) == 1 else imgs


@torch.no_grad()
def eval_episode(model, ds, action_space, policy, device):
    """Replay one single-episode dataset in time order; return a filled ActionRecorder.

    reset() then sequential predict() reproduces exactly how deploy.py drives the
    policy (incl. ACT's temporal ensembling), so the predicted stream matches the robot
    path that this checkpoint would command for these observations.
    """
    model.reset()
    rec = ActionRecorder(action_space=action_space, policy=policy, source="eval")
    for i in range(len(ds)):
        sample = ds[i]
        state = sample["observation.state"].unsqueeze(0).to(device)
        imgs = _to_model_images(
            {k: v for k, v in sample.items() if k.startswith("observation.images.")},
            device)
        task = [sample["task"]] if sample.get("task") else None

        action = model.predict(state, imgs, task=task)
        if action.dim() == 2:
            action = action[0]
        action = action.cpu().numpy()

        gt = sample["action"][0].numpy()        # the action to take NOW (chunk step 0)
        rec.add(t=i, state=sample["observation.state"].numpy(),
                pred=action, gt=gt, gripper=float(action[-1]))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint (best.pt / epoch_XXXX.pt)")
    ap.add_argument("--episodes", nargs="+", type=int, default=None,
                    help="how many test episodes (single int) or which ones (list of "
                         "indices). default: first 2 held-out episodes")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap frames per episode (default: whole episode)")
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt_dir>/eval)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    policy = ck.get("policy", "act")
    action_space = ck.get("action_space", "joint")
    model = _load_policy_module(policy).build_model(cfg, ck["stats"], device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"loaded {policy}  epoch={ck.get('epoch')}  action_space={action_space}")

    d, t = cfg["dataset"], cfg["training"]
    stride = d.get("frame_stride", 1)
    n_eps = int(FR5Dataset(d["root"], frame_stride=stride).info["total_episodes"])
    _, val_eps = FR5Dataset.episode_split(n_eps, d["val_frac"], t["seed"])
    print(f"held-out test episodes: {val_eps}")

    # resolve which episodes to evaluate
    if args.episodes is None:
        chosen = val_eps[:2]
    elif len(args.episodes) == 1:
        chosen = val_eps[:args.episodes[0]]
    else:
        chosen = [e for e in args.episodes if e in val_eps]
        skipped = [e for e in args.episodes if e not in val_eps]
        if skipped:
            print(f"[warn] ignoring {skipped} — not in the held-out test split")
    if not chosen:
        raise SystemExit("no valid test episodes selected")

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "eval"
    print(f"evaluating episodes {chosen} -> {out_dir}\n")

    all_mae = []
    for ep in chosen:
        ds = FR5Dataset(d["root"], d["chunk_size"], d["use_image"],
                        tuple(d["image_size"]), episode_indices=[ep],
                        aug_level="none", frame_stride=stride)
        if args.max_steps:
            ds._samples = ds._samples[:args.max_steps]
        if len(ds) == 0:
            print(f"[skip] episode {ep}: no frames")
            continue
        rec = eval_episode(model, ds, action_space, policy, device)
        path = rec.save(out_dir / f"ep{ep:03d}.npz")
        print(f"episode {ep}: {len(rec)} steps -> {path}")
        plot_record(path)

        arrays, _ = load_record(path)
        mae = np.nanmean(np.abs(arrays["pred"] - arrays["gt"]))
        all_mae.append(mae)
        print(f"  episode {ep} overall MAE = {mae:.4f}\n")

    if all_mae:
        print(f"mean overall MAE across {len(all_mae)} episode(s): {np.mean(all_mae):.4f}")


if __name__ == "__main__":
    main()
