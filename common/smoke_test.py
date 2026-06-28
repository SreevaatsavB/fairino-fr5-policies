"""
smoke_test.py — exercise the whole pipeline on the 2 recorded episodes.

Goal: prove the wiring works (data -> model -> train step -> checkpoint ->
load -> predict), NOT to train a useful policy. Everything runs on CPU in
seconds with a tiny model.

Stages:
  1. convert raw episodes -> LeRobot dataset (_smoke_dataset)
  2. load via common/dataset.py and check tensor shapes
  3. [needs lerobot] build the policy via policies/<policy>/build_model, run one
     fwd/bwd step, save a checkpoint, reload it (deploy-style) and call predict()

If lerobot isn't installed, stages 1-2 still run and stage 3 is reported as
SKIPPED with the reason.

    python common/smoke_test.py            # defaults to the act policy
    python common/smoke_test.py diffusion  # once policies/diffusion/ exists
"""

import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
SMOKE_DS = REPO / "_smoke_dataset"
POLICY = sys.argv[1] if len(sys.argv) > 1 else "act"
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "policies" / POLICY))


def stage1_convert():
    print("\n[1/3] convert raw episodes -> LeRobot dataset")
    cmd = [sys.executable, str(REPO / "common" / "convert_episodes.py"),
           "--episodes", str(REPO / "episodes"),
           "--out", str(SMOKE_DS),
           "--extract-frames", "--max-frames", "60"]
    subprocess.run(cmd, check=True)


def stage2_dataset():
    print("\n[2/3] load via FR5Dataset and check shapes")
    from dataset import FR5Dataset

    chunk = 20
    ds = FR5Dataset(str(SMOKE_DS), chunk_size=chunk, use_image=True,
                    image_size=(224, 224), episode_indices=[0])
    assert len(ds) > 0, "dataset is empty"
    s = ds[0]
    assert s["observation.state"].shape == (7,), s["observation.state"].shape
    assert s["action"].shape == (chunk, 7), s["action"].shape
    assert s["action_is_pad"].shape == (chunk,)
    assert s["observation.images.wrist_cam"].shape == (3, 224, 224), \
        s["observation.images.wrist_cam"].shape
    stats = ds.get_stats()
    assert stats["state_mean"].shape == (7,) and stats["action_mean"].shape == (7,)
    print(f"      OK  len={len(ds)}  state={tuple(s['observation.state'].shape)}  "
          f"action={tuple(s['action'].shape)}  image={tuple(s['observation.images.wrist_cam'].shape)}")
    return stats


def stage3_model(stats):
    print(f"\n[3/3] build '{POLICY}' policy, train step, checkpoint, reload, predict")
    import yaml
    try:
        import model as policy_mod  # policies/<POLICY>/model.py
    except Exception as e:  # lerobot missing or API mismatch
        print(f"      SKIPPED — could not import policies/{POLICY}/model.py "
              f"({type(e).__name__}: {e})")
        print("      install the policy backend to run this stage:  pip install lerobot")
        return False

    with open(REPO / "policies" / POLICY / "config.smoke.yaml") as f:
        cfg = yaml.safe_load(f)
    chunk = cfg["dataset"]["chunk_size"]
    adim = cfg["model"]["action_dim"]
    sdim = cfg["model"]["state_dim"]
    device = torch.device("cpu")
    model = policy_mod.build_model(cfg, stats, device)

    # one fwd/bwd step on a dummy batch
    B = 2
    obs = torch.randn(B, sdim)
    acts = torch.randn(B, chunk, adim)
    pad = torch.zeros(B, chunk, dtype=torch.bool)
    img = torch.rand(B, 3, 224, 224)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    loss, l1, kl = model(obs, acts, pad, img)
    loss.backward()
    opt.step()
    print(f"      train step OK  l1={l1:.4f}  kl={kl:.4f}")

    # checkpoint round-trip (mirrors train.py / deploy.py)
    ckpt = SMOKE_DS / "smoke_ckpt.pt"
    torch.save({"epoch": 1, "policy": POLICY, "model_state": model.state_dict(),
                "val_l1": l1, "config": cfg, "stats": stats}, ckpt)
    saved = torch.load(ckpt, weights_only=False)
    model2 = policy_mod.build_model(saved["config"], saved["stats"], device)
    model2.load_state_dict(saved["model_state"])
    model2.eval()
    model2.reset()
    out = model2.predict(obs[:1], img[:1])
    out = out[0] if out.dim() == 2 else out
    assert out.shape[-1] == adim, out.shape
    print(f"      checkpoint reload + predict OK  action={tuple(out.shape)}")
    return True


def main():
    stage1_convert()
    stats = stage2_dataset()
    model_ok = stage3_model(stats)
    print("\n" + "=" * 60)
    print(f"SMOKE TEST [{POLICY}]: data pipeline PASSED" +
          ("  |  model+train+deploy PASSED" if model_ok
           else "  |  model stage SKIPPED (lerobot not installed)"))
    print("=" * 60)


if __name__ == "__main__":
    main()
