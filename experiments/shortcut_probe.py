#!/usr/bin/env python3
"""
shortcut_probe.py — does the policy use the CAMERAS, or just its own joint STATE?

This is the causal-intervention probe referenced in docs/proprioception_modes.md and
docs/failure_analysis/. It quantifies the *proprioceptive shortcut*: how much the predicted
action depends on the cameras vs the joint state. The deployed DINOv2 ACT had
state-sensitivity ~2.2x image-sensitivity (object-blind); a healthy, vision-using model
should be IMAGE-dominant (ratio < 1).

For each validation sample we take the predicted action chunk (model.predict_chunk, which
bypasses temporal ensembling) under input perturbations and measure how much it MOVES
(mean |Δaction|, in real units):

  • zero-images    cameras blanked (state real)        -> image_sensitivity (presence)
  • shuffle-images state paired with ANOTHER sample's  -> imagecontent_sensitivity (content)
                   images (the conflict-swap)
  • zero-state     joint state zeroed (images real)    -> state_sensitivity

Plus reference scales:
  • output_spread  std of the predictions across inputs; ~0 => a canned/constant trajectory
  • action_std(GT) std of the ground-truth actions (the signal to be matched)

Headline metric:
  state/image ratio = state_sensitivity / image_sensitivity
     >> 1  => object-blind (proprioceptive shortcut)  |  < 1 => image-dominant (good)

Usage:
  # one checkpoint
  python experiments/shortcut_probe.py --ckpt policies/act/checkpoints_joint_dino/best.pt
  # rank several checkpoints and recommend the most image-dominant (model SELECTION,
  # replacing val_l1 which is anti-correlated with deploy success — see failure_analysis 03)
  python experiments/shortcut_probe.py --checkpoints a/best.pt b/best.pt c/best.pt
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "common"))
from dataset import FR5Dataset   # noqa: E402


def _load_policy_module(policy):
    spec = importlib.util.spec_from_file_location(
        f"policy_{policy}", REPO / "policies" / policy / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def probe(ckpt_path, n_batches, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    from vla_pretrained import strip_pretrained_for_checkpoint
    strip_pretrained_for_checkpoint(cfg)   # skip 6 GB base re-download (VLA policies)
    model = _load_policy_module(ck.get("policy", "act")).build_model(cfg, ck["stats"], device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    if not hasattr(model, "predict_chunk"):
        raise SystemExit(f"{ck.get('policy')} has no predict_chunk(); the probe needs it "
                         "(ACT exposes it). Add predict_chunk to the wrapper to probe it.")

    d, t = cfg["dataset"], cfg["training"]
    stride = d.get("frame_stride", 1)
    n_eps = int(FR5Dataset(d["root"], frame_stride=stride).info["total_episodes"])
    _, val = FR5Dataset.episode_split(n_eps, d["val_frac"], t["seed"])
    ds = FR5Dataset(d["root"], d["chunk_size"], d["use_image"], tuple(d["image_size"]),
                    episode_indices=val, aug_level="none", frame_stride=stride)
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, num_workers=2)

    def chunk(state, imgs):
        return model.predict_chunk(state, imgs).cpu().numpy()

    d_zeroimg = d_shufimg = d_zerostate = 0.0
    preds, cnt = [], 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        state = batch["observation.state"].to(device)
        imgs = {k: v.to(device) for k, v in batch.items()
                if k.startswith("observation.images.")}
        B = state.shape[0]
        if B < 2:
            continue
        real      = chunk(state, imgs)
        zeroimg   = chunk(state, {k: torch.zeros_like(v) for k, v in imgs.items()})
        roll      = torch.roll(torch.arange(B), 1)            # conflict-swap: state_A + imgs_B
        shufimg   = chunk(state, {k: v[roll] for k, v in imgs.items()})
        zerostate = chunk(torch.zeros_like(state), imgs)
        d_zeroimg   += np.abs(real - zeroimg).mean()   * B
        d_shufimg   += np.abs(real - shufimg).mean()   * B
        d_zerostate += np.abs(real - zerostate).mean() * B
        preds.append(real.reshape(real.shape[0], -1))
        cnt += B

    if cnt == 0:
        raise SystemExit("no usable validation batches (need batch_size >= 2)")
    preds = np.concatenate(preds, 0)
    img_s   = d_zeroimg / cnt
    state_s = d_zerostate / cnt
    return {
        "ckpt":                str(ckpt_path),
        "proprio_mode":        cfg["model"].get("proprio_mode"),
        "val_l1":              float(ck.get("val_l1", float("nan"))),
        "image_sensitivity":   float(img_s),
        "imagecontent_sens":   float(d_shufimg / cnt),
        "state_sensitivity":   float(state_s),
        "state_over_image":    float(state_s / max(img_s, 1e-6)),
        "output_spread":       float(preds.std(0).mean()),
        "action_std_GT":       float(np.array(ds.get_stats()["action_std"]).mean()),
        "n_samples":           cnt,
    }


def _verdict(r):
    img, out, ratio = r["image_sensitivity"], r["output_spread"], r["state_over_image"]
    if out < 1e-3:
        return "CONSTANT output (canned trajectory — ignores all inputs)"
    if img < 1e-3 or img < 0.02 * out:
        return "IGNORES vision (object-blind proprioceptive shortcut)"
    if ratio > 1.5:
        return f"state-dominant ({ratio:.1f}x) — leans on proprioception"
    if ratio > 0.9:
        return f"balanced ({ratio:.1f}x)"
    return f"IMAGE-dominant ({ratio:.1f}x) — uses the cameras (good)"


def _print_one(r):
    print(f"\n=== vision probe: {r['ckpt']}  (proprio={r['proprio_mode']}, n={r['n_samples']}) ===")
    print(f"  image sensitivity    |Δa| cameras ZEROED   : {r['image_sensitivity']:.4f}")
    print(f"  image-content sens.  |Δa| cameras SHUFFLED : {r['imagecontent_sens']:.4f}")
    print(f"  state sensitivity    |Δa| STATE zeroed     : {r['state_sensitivity']:.4f}")
    print(f"  state / image ratio  (>1 = object-blind)   : {r['state_over_image']:.2f}x")
    print(f"  output spread        std over inputs       : {r['output_spread']:.4f}")
    print(f"  action std (GT)      target signal scale   : {r['action_std_GT']:.4f}")
    print(f"  val_l1 (NOT a good selector)               : {r['val_l1']:.4f}")
    print(f"  --> {_verdict(r)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="single checkpoint to probe")
    ap.add_argument("--checkpoints", nargs="+", help="rank several checkpoints (model selection)")
    ap.add_argument("--n-batches", type=int, default=60)
    args = ap.parse_args()
    if not args.ckpt and not args.checkpoints:
        ap.error("give --ckpt <p> or --checkpoints <p1> <p2> ...")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    if args.ckpt:
        _print_one(probe(args.ckpt, args.n_batches, device))
        return

    # selection mode: probe each, rank by image-dominance (lowest state/image ratio with
    # non-trivial image sensitivity), print a table, recommend the winner.
    results = []
    for p in args.checkpoints:
        try:
            results.append(probe(p, args.n_batches, device))
        except Exception as e:                                   # keep going if one fails
            print(f"[skip] {p}: {type(e).__name__}: {e}")
    if not results:
        raise SystemExit("no checkpoints probed successfully")
    results.sort(key=lambda r: r["state_over_image"])            # most image-dominant first
    print(f"\n{'='*92}\nMODEL SELECTION — ranked by vision-dominance (state/image ratio, lower=better)\n{'='*92}")
    print(f"{'rank':>4}  {'state/img':>9}  {'img_sens':>8}  {'state_sens':>10}  {'val_l1':>7}  ckpt")
    for i, r in enumerate(results, 1):
        print(f"{i:>4}  {r['state_over_image']:>9.2f}  {r['image_sensitivity']:>8.4f}  "
              f"{r['state_sensitivity']:>10.4f}  {r['val_l1']:>7.4f}  {r['ckpt']}")
    best = results[0]
    print(f"\nRECOMMENDED: {best['ckpt']}")
    print(f"  {_verdict(best)}")
    print("  (selection by vision-dominance, NOT val_l1 — see docs/failure_analysis/03.)")


if __name__ == "__main__":
    main()
