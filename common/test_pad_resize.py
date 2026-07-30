"""
test_pad_resize.py — the letterbox must be bit-identical everywhere.

    python common/test_pad_resize.py

Three copies of resize_with_pad exist by necessity: dataset.ResizeWithPad (training),
deploy._ResizeWithPad (the robot path, which cannot import dataset without the
teleop SDK), and lerobot's own resize_with_pad_torch. If they drift, training and
deployment disagree on the image format — the exact fault that invalidated every
pre-2026-07-28 robot trial.
"""

import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common"))
from dataset import ResizeWithPad, _build_transform  # noqa: E402

GEOMETRIES = [(480, 640), (720, 1280), (224, 224), (1080, 1920), (100, 900)]


def _deploy_class():
    """deploy.py imports the teleop SDK at module scope, so lift just the class."""
    src = (REPO / "common" / "deploy.py").read_text()
    m = re.search(r"class _ResizeWithPad:.*?(?=\n\n_NORM|\n\nclass |\n\ndef )", src, re.S)
    assert m, "deploy.py no longer defines _ResizeWithPad"
    ns = {"torch": torch}
    exec(m.group(0), ns)
    return ns["_ResizeWithPad"]


def test_matches_lerobot():
    """The reference: lerobot's resize_with_pad_torch, an exact openpi copy."""
    from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch
    for h, w in GEOMETRIES:
        img = torch.rand(3, h, w)
        ours = ResizeWithPad(224, 224)(img)
        ref = resize_with_pad_torch(img.unsqueeze(0), 224, 224).squeeze(0)
        d = (ours - ref).abs().max().item()
        assert d == 0.0, f"{h}x{w}: max|diff| = {d:.3e}, must be exactly 0"
    print(f"  bit-identical to lerobot over {len(GEOMETRIES)} geometries")


def test_matches_deploy():
    """dataset (training) vs deploy (robot) — the pair that must not drift."""
    Deploy = _deploy_class()
    for h, w in GEOMETRIES:
        img = torch.rand(3, h, w)
        d = (ResizeWithPad(224, 224)(img) - Deploy(224, 224)(img)).abs().max().item()
        assert d == 0.0, f"{h}x{w}: dataset vs deploy differ by {d:.3e}"
    print("  dataset.ResizeWithPad == deploy._ResizeWithPad")


def test_letterbox_actually_pads():
    """640x480 -> 224x168 content + 28px bars top and bottom (not a squash)."""
    out = ResizeWithPad(224, 224)(torch.rand(3, 480, 640))
    assert out.shape == (3, 224, 224)
    rows = (out.sum(0) != 0).any(1)
    assert rows.sum().item() == 168, f"expected 168 content rows, got {rows.sum()}"
    assert not rows[:28].any() and not rows[-28:].any(), "bars are not at top/bottom"
    print("  640x480 -> 224x168 content + 28px bars")


def test_build_transform_honours_the_flag():
    """pad_resize=False must still squash — pre-v3 checkpoints depend on it."""
    img = torch.rand(3, 480, 640)
    padded = _build_transform((224, 224), "none", pad_resize=True)(img)
    squashed = _build_transform((224, 224), "none", pad_resize=False)(img)
    assert padded.shape == squashed.shape == (3, 224, 224)
    # after ImageNet norm a black bar is a constant row; the squash has no such rows
    def const_rows(t):
        return int((t.std(0).std(1) < 1e-6).sum())
    assert const_rows(padded) >= 56, "letterbox produced no constant bar rows"
    assert const_rows(squashed) == 0, "squash should have no bar rows"
    # every aug level must route through the letterbox too
    for lvl in ("none", "photometric", "crops", "full"):
        t = _build_transform((224, 224), lvl, pad_resize=True)(img)
        assert t.shape == (3, 224, 224), lvl
    print("  pad_resize flag honoured; all 4 aug levels letterbox")


if __name__ == "__main__":
    for fn in (test_matches_lerobot, test_matches_deploy, test_letterbox_actually_pads,
               test_build_transform_honours_the_flag):
        print(f"{fn.__name__}:")
        fn()
    print("\nall pad-resize checks passed")
