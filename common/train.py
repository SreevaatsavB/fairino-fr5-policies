import argparse
import csv
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import FR5Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_info():
    """Best-effort git provenance so a checkpoint can be traced to exact source."""
    def _run(*a):
        try:
            return subprocess.check_output(a, cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain")),  # uncommitted changes present
    }


def write_run_info(ckpt_dir, args, cfg, config_path, device, model,
                   train_loader, val_loader, train_ds, stats):
    """Snapshot everything needed to reproduce / error-analyse this run.

    Written once at start to <ckpt_dir>/run_info.json. Pairs with metrics.csv
    (per-epoch curves) so any later deploy failure can be tied back to the exact
    code, config, data split, and environment that produced the checkpoint.
    """
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "policy":      args.policy,
        "config_path": str(config_path),
        "argv":        sys.argv,
        "git":         _git_info(),
        "device":      str(device),
        "torch":       torch.__version__,
        "python":      platform.python_version(),
        "platform":    platform.platform(),
        "trainable_params": n_train,
        "data": {
            "root":            cfg["dataset"]["root"],
            "action_space":    train_ds.info.get("action_space", "joint"),
            "total_episodes":  int(train_ds.info.get("total_episodes", 0)),
            "train_frames":    len(train_loader.dataset),
            "val_frames":      len(val_loader.dataset),
            "frame_stride":    cfg["dataset"].get("frame_stride", 1),
            "image_size":      cfg["dataset"].get("image_size"),
            "use_image":       cfg["dataset"].get("use_image"),
        },
        # stats are JSON-unfriendly tensors; record shapes + mean/std summaries only
        "stats_keys":  sorted(stats.keys()) if isinstance(stats, dict) else None,
        "config":      cfg,   # full resolved config (post proprio overrides)
    }
    path = Path(ckpt_dir) / "run_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"run info -> {path}")


def load_policy_module(policy: str):
    """Import policies/<policy>/model.py, which must expose
    build_model(cfg: dict, stats: dict, device) -> nn.Module."""
    path = REPO_ROOT / "policies" / policy / "model.py"
    if not path.exists():
        raise SystemExit(f"unknown policy '{policy}': {path} not found")
    spec = importlib.util.spec_from_file_location(f"policy_{policy}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_model"):
        raise SystemExit(f"policies/{policy}/model.py must define build_model(cfg, stats, device)")
    return mod


def get_device(cfg):
    pref = cfg["training"]["device"]
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(cfg):
    d = cfg["dataset"]
    t = cfg["training"]

    aug = d.get("aug_level", "none")

    # split by episode index to avoid leakage
    tmp = FR5Dataset(d["root"], chunk_size=d["chunk_size"],
                     use_image=d["use_image"], image_size=tuple(d["image_size"]))
    n_total = int(tmp.info["total_episodes"])
    train_eps, val_eps = FR5Dataset.episode_split(n_total, d["val_frac"], t["seed"])

    stride = d.get("frame_stride", 1)
    train_ds = FR5Dataset(d["root"], d["chunk_size"], d["use_image"],
                          tuple(d["image_size"]), episode_indices=train_eps,
                          aug_level=aug, frame_stride=stride)
    val_ds   = FR5Dataset(d["root"], d["chunk_size"], d["use_image"],
                          tuple(d["image_size"]), episode_indices=val_eps,
                          aug_level="none", frame_stride=stride)   # never augment val

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=t["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"train episodes={len(train_eps)}  frames={len(train_ds)}")
    print(f"val   episodes={len(val_eps)}    frames={len(val_ds)}")
    return train_loader, val_loader, train_ds


def build_model(policy_mod, cfg, stats, device):
    model = policy_mod.build_model(cfg, stats, device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n/1e6:.1f}M")
    return model


def run_epoch(model, loader, optimizer, cfg, device, train=True):
    model.train(train)
    clip  = cfg["training"]["grad_clip"]
    log_n = cfg["training"]["log_every"]

    total_l1 = total_kl = total_gn = 0.0

    # lerobot's ACTPolicy.forward already folds kl_weight into `loss`, so we
    # backprop it directly; l1 / kl come back as floats for logging only.
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            obs   = batch["observation.state"].to(device)
            acts  = batch["action"].to(device)
            pad   = batch["action_is_pad"].to(device)
            # collect all camera streams (observation.images.*). Pass a single
            # tensor when there is ONE camera (back-compat for the single-cam
            # policies) and a {key: tensor} dict when there are several (e.g. the
            # 2-camera DINOv2 ACT / 2-camera pi0). Both consume the dict form via
            # their own camera_names/image_keys wiring.
            imgs = {k: v.to(device) for k, v in batch.items()
                    if k.startswith("observation.images.")}
            if len(imgs) == 1:
                img = next(iter(imgs.values()))
            elif len(imgs) > 1:
                img = imgs
            else:
                img = None
            task  = batch.get("task")  # list[str] or None; used by language-conditioned policies

            loss, l1, kl = model(obs, acts, pad, img, task=task)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_ returns the total grad norm *before* clipping —
                # log it as an instability signal (a sudden spike precedes divergence).
                gn = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                total_gn += float(gn)

            total_l1 += l1
            total_kl += kl

            if train and (step + 1) % log_n == 0:
                print(f"    step {step+1}/{len(loader)}  "
                      f"l1={l1:.4f}  kl={kl:.4f}")

    n = len(loader)
    return total_l1 / n, total_kl / n, (total_gn / n if train else float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="act",
                        help="policy under policies/<name>/ (default: act)")
    parser.add_argument("--config", default=None,
                        help="config yaml (default: policies/<policy>/config.yaml)")
    parser.add_argument("--proprio", choices=["full", "dropout", "none"], default=None,
                        help="override model.proprio_mode for a benchmark run; "
                             "checkpoints are isolated under <ckpt_dir>/proprio_<mode>")
    parser.add_argument("--proprio-rate", type=float, default=None,
                        help="override model.proprio_dropout_rate (dropout mode only)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else \
        REPO_ROOT / "policies" / args.policy / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # proprio benchmark overrides — isolate checkpoints so variants don't clobber
    if args.proprio is not None:
        cfg["model"]["proprio_mode"] = args.proprio
        base = Path(cfg["training"]["checkpoint_dir"])
        cfg["training"]["checkpoint_dir"] = str(base / f"proprio_{args.proprio}")
    if args.proprio_rate is not None:
        cfg["model"]["proprio_dropout_rate"] = args.proprio_rate

    policy_mod = load_policy_module(args.policy)
    print(f"policy: {args.policy}  config: {config_path}")

    torch.manual_seed(cfg["training"]["seed"])
    device = get_device(cfg)
    print(f"device: {device}")

    train_loader, val_loader, train_ds = build_loaders(cfg)
    stats = train_ds.get_stats()
    model = build_model(policy_mod, cfg, stats, device)

    # Param groups: a fine-tuned DINOv2 backbone (...backbone.dino.*) trains at 0.1x
    # LR so the pretrained features adapt gently on a small dataset; the from-scratch
    # head gets the full LR. With a fully-frozen backbone there are no dino params
    # with grad, so that group is simply absent (identical to plain AdamW).
    base_lr = cfg["training"]["lr"]
    dino_params  = [p for n, p in model.named_parameters()
                    if p.requires_grad and "backbone.dino" in n]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and "backbone.dino" not in n]
    groups = [{"params": other_params, "lr": base_lr}]
    if dino_params:
        groups.append({"params": dino_params, "lr": base_lr * 0.1})
        print(f"optimizer: head @ lr={base_lr:g}, DINOv2 fine-tune "
              f"({len(dino_params)} tensors) @ lr={base_lr * 0.1:g}")
    optimizer = torch.optim.AdamW(
        groups, lr=base_lr, weight_decay=cfg["training"]["weight_decay"],
    )

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # per-epoch metrics log (plot with tools/plot_metrics.py). kl_weight_eff is the
    # *annealed* KL weight at epoch end (ACT only) so the KL curve is interpretable.
    metrics_path = ckpt_dir / "metrics.csv"
    metrics_header = ["epoch", "train_l1", "train_kl", "val_l1", "val_kl",
                      "train_val_gap", "grad_norm", "kl_weight_eff", "lr", "seconds"]
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(metrics_header)
    print(f"metrics -> {metrics_path}")

    # one-time provenance snapshot for reproducibility / error analysis
    write_run_info(ckpt_dir, args, cfg, config_path, device, model,
                   train_loader, val_loader, train_ds, stats)

    best_val = float("inf")
    max_ep   = cfg["training"]["max_epochs"]
    save_n   = cfg["training"]["save_every"]

    for epoch in range(1, max_ep + 1):
        t0 = time.time()

        print(f"\nepoch {epoch}/{max_ep}")
        train_l1, train_kl, grad_norm = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        val_l1,   val_kl,   _         = run_epoch(model, val_loader,   optimizer, cfg, device, train=False)

        elapsed = time.time() - t0
        print(f"  train l1={train_l1:.4f}  kl={train_kl:.4f}  "
              f"val l1={val_l1:.4f}  gap={val_l1 - train_l1:+.4f}  "
              f"|g|={grad_norm:.2f}  ({elapsed:.0f}s)")

        # append the epoch's metrics to the CSV. kl_weight_eff = the annealed CVAE
        # weight at this point (ACT only; blank for policies without the schedule).
        # val_kl, grad_norm, and the train/val gap are logged for error analysis:
        # val_kl rising while train_kl stays low = latent overfitting; a grad_norm
        # spike precedes divergence; gap is the overfitting signal (doc 09).
        kl_eff = ""
        if all(hasattr(model, a) for a in ("_kl_step", "kl_weight", "kl_warmup_steps")):
            kl_eff = f"{model.kl_weight * min(1.0, float(model._kl_step.item()) / model.kl_warmup_steps):.4f}"
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_l1:.6f}", f"{train_kl:.6f}",
                                    f"{val_l1:.6f}", f"{val_kl:.6f}",
                                    f"{val_l1 - train_l1:.6f}", f"{grad_norm:.4f}",
                                    kl_eff, base_lr, f"{elapsed:.1f}"])

        is_best = val_l1 < best_val
        if is_best:
            best_val = val_l1

        if epoch % save_n == 0 or is_best:
            ckpt = {
                "epoch":          epoch,
                "policy":         args.policy,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_l1":         val_l1,
                "config":         cfg,
                "stats":          stats,
                # how to execute this model's actions on the robot (joint | delta_eef);
                # stamped by convert_episodes into the dataset, deploy reads it back.
                "action_space":   train_ds.info.get("action_space", "joint"),
            }
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(ckpt, path)
            if is_best:
                torch.save(ckpt, ckpt_dir / "best.pt")
                print(f"  ↑ best  saved → {ckpt_dir / 'best.pt'}")
            else:
                print(f"  saved → {path}")

    print(f"\ndone.  best val l1={best_val:.4f}")


if __name__ == "__main__":
    main()
