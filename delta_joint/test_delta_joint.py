"""
test_delta_joint.py — self-check for the action-scale fixes.

    python delta_joint/test_delta_joint.py

Runs against _smoke_dataset/. Fails if either transform, its inverse, or the
recomputed stats break. No framework, no fixtures.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "common"), str(Path(__file__).resolve().parent)]

from dataset import FR5Dataset          # noqa: E402
from dataset_delta import (DeltaJointDataset, JOINTS, absolutise, hold,  # noqa: E402
                           for_inference, encode_action_space, parse_action_space)

ROOT, CHUNK, STRIDE = str(REPO / "_smoke_dataset"), 8, 3


def _ds(**kw):
    return DeltaJointDataset(ROOT, chunk_size=CHUNK, use_image=False, **kw)


def test_roundtrip():
    """delta + state == the original absolute action, exactly, on every sample."""
    plain = FR5Dataset(ROOT, chunk_size=CHUNK, use_image=False)
    delta = _ds(action_stride=1)
    assert len(plain) == len(delta) > 0
    for i in range(0, len(plain), 7):
        a, d = plain[i], delta[i]
        torch.testing.assert_close(a["observation.state"], d["observation.state"])
        restored = d["action"].clone()
        restored[:, JOINTS] += a["observation.state"][JOINTS]
        torch.testing.assert_close(restored, a["action"])
        # gripper is a command, not a position — it must pass through untouched
        torch.testing.assert_close(d["action"][:, 6], a["action"][:, 6])
    print(f"  roundtrip exact over {len(range(0, len(plain), 7))} samples")


def test_stride_picks_every_kth_frame():
    """The strided chunk must be the same rows the stock chunk would give at 0,
    K, 2K... — off-by-one here silently trains on the wrong future."""
    plain = FR5Dataset(ROOT, chunk_size=CHUNK * STRIDE, use_image=False)
    strided = _ds(action_stride=STRIDE, delta=False)
    _, frame_abs, _, ep_to = strided._samples[0]
    assert frame_abs + CHUNK * STRIDE <= ep_to, "smoke episode too short for the test"
    torch.testing.assert_close(strided[0]["action"], plain[0]["action"][::STRIDE])
    print(f"  stride {STRIDE} chunk == stock chunk[::{STRIDE}]")


def test_stride_extends_the_horizon():
    """The point of striding: a bigger step, over a longer window."""
    s1 = _ds(action_stride=1).get_stats()["action_std"][:6].mean()
    sk = _ds(action_stride=STRIDE).get_stats()["action_std"][:6].mean()
    assert sk > s1, f"stride {STRIDE} std {sk} should exceed stride 1 std {s1}"
    print(f"  delta action_std: stride 1 {s1:.4f} -> stride {STRIDE} {sk:.4f} deg")


def test_stats_are_on_the_transformed_actions():
    """The whole point: the normaliser must shrink, or nothing was gained."""
    plain = FR5Dataset(ROOT, chunk_size=CHUNK, use_image=False).get_stats()
    delta = _ds(action_stride=1).get_stats()
    np.testing.assert_allclose(plain["state_mean"], delta["state_mean"])  # untouched
    a, d = plain["action_std"][:6].mean(), delta["action_std"][:6].mean()
    assert d < a, f"delta std {d} not below absolute {a}"
    np.testing.assert_allclose(plain["action_std"][6], delta["action_std"][6], rtol=1e-5)
    print(f"  action_std joints: absolute {a:.3f} -> delta {d:.3f} deg ({a / d:.1f}x tighter)")


def test_stats_match_getitem():
    """get_stats() must describe the tensors __getitem__ actually emits, padding
    included — a mismatch silently mis-normalises every target."""
    for stride in (1, STRIDE):
        ds = _ds(action_stride=stride)
        # float64: np.std in float32 accumulates ~2e-3 error over the real
        # dataset's 270k chunk-entries (measured) — get_stats itself is exact
        seen = np.concatenate([ds[i]["action"].numpy() for i in range(len(ds))]).astype(np.float64)
        st = ds.get_stats()
        np.testing.assert_allclose(seen.mean(0), st["action_mean"], atol=1e-3)
        np.testing.assert_allclose(seen.std(0).clip(1e-6), st["action_std"], atol=1e-3)
    print(f"  get_stats matches __getitem__ at stride 1 and {STRIDE} (incl. padding)")


def test_absolutise_inverts():
    """The inference wrapper must undo exactly what the dataset did."""
    plain = FR5Dataset(ROOT, chunk_size=CHUNK, use_image=False)
    delta = _ds(action_stride=1)
    want, got_delta = plain[0]["action"][0], delta[0]["action"][0]
    obs = plain[0]["observation.state"][None]        # (1, 7) as deploy passes it

    class Stub:
        def predict(self, obs_state, obs_image=None, task=None):
            return got_delta[None].clone()           # (1, 7)
        def reset(self): pass

    torch.testing.assert_close(absolutise(Stub()).predict(obs)[0], want)

    class Stub1D(Stub):                              # policies returning (7,)
        def predict(self, obs_state, obs_image=None, task=None):
            return got_delta.clone()

    torch.testing.assert_close(absolutise(Stub1D()).predict(obs), want)
    print("  absolutise() inverts the transform for (1,7) and (7,) outputs")


def test_hold_repeats_each_action():
    """Without this the arm replays the trajectory `stride` times too fast."""
    calls = {"n": 0}

    class Stub:
        def predict(self, obs_state, obs_image=None, task=None):
            calls["n"] += 1
            return torch.full((7,), float(calls["n"]))
        def reset(self): calls["n"] = 0

    m = hold(Stub(), STRIDE)
    out = [m.predict(torch.zeros(1, 7))[0].item() for _ in range(STRIDE * 3)]
    assert out == [1.0] * STRIDE + [2.0] * STRIDE + [3.0] * STRIDE, out
    assert calls["n"] == 3, f"model queried {calls['n']} times, expected 3"
    m.reset()                                        # must clear the held action
    assert m.predict(torch.zeros(1, 7))[0].item() == 1.0
    print(f"  hold({STRIDE}) repeats each action {STRIDE}x and reset() clears it")


def test_action_space_string_roundtrips():
    """deploy reads the whole recipe out of this one string."""
    for delta in (True, False):
        for stride in (1, STRIDE):
            s = encode_action_space(delta, stride)
            base, k = parse_action_space(s)
            assert base == ("delta_joint" if delta else "joint") and k == stride, s
    assert _ds(action_stride=STRIDE).info["action_space"] == f"delta_joint@{STRIDE}"
    assert parse_action_space("joint") == ("joint", 1)          # stock checkpoints
    assert parse_action_space("delta_eef") == ("delta_eef", 1)
    print("  action_space string encodes/parses delta + stride")


def test_for_inference_applies_both():
    """End to end: a delta_joint@K checkpoint must come out absolute AND held."""
    plain = FR5Dataset(ROOT, chunk_size=CHUNK, use_image=False)
    delta = _ds(action_stride=STRIDE)
    obs = plain[0]["observation.state"][None]
    d0 = delta[0]["action"][0]

    class Stub:
        def predict(self, obs_state, obs_image=None, task=None):
            return d0[None].clone()
        def reset(self): pass

    m = for_inference(Stub(), f"delta_joint@{STRIDE}")
    outs = [m.predict(obs)[0].clone() for _ in range(STRIDE)]
    for o in outs:
        torch.testing.assert_close(o, plain[0]["action"][0])     # absolute again
    print(f"  for_inference('delta_joint@{STRIDE}') = absolutise + hold")


def test_for_inference_is_idempotent():
    """Applying the wrappers twice must NOT add the state twice — that would be a
    silent ~13 deg offset on every joint command. eval wraps once per episode."""
    class Stub:
        def predict(self, obs_state, obs_image=None, task=None):
            return torch.zeros(7)
        def reset(self): pass

    m = Stub()
    for _ in range(3):
        for_inference(m, f"delta_joint@{STRIDE}")
    obs = torch.full((1, 7), 10.0)
    torch.testing.assert_close(m.predict(obs)[JOINTS], torch.full((6,), 10.0))
    print("  for_inference applied 3x still adds the state exactly once")


def test_held_actions_are_independent():
    """hold() hands the same cached action out `stride` times; a caller editing
    one copy in place must not corrupt the rest."""
    class Stub:
        def predict(self, obs_state, obs_image=None, task=None):
            return torch.ones(7)
        def reset(self): pass

    m = hold(Stub(), STRIDE)
    first = m.predict(torch.zeros(1, 7))
    first[0] = 999.0
    assert m.predict(torch.zeros(1, 7))[0].item() == 1.0, "held action was aliased"
    print("  held actions are independent copies")


def test_take_flag_edges():
    """A missing flag value must fail loudly, not IndexError deep in argparse."""
    from run import _take_flag
    assert _take_flag(["--policy", "pi0"], "--action-stride", "5")[0] == "5"
    assert _take_flag(["--action-stride", "7"], "--action-stride", "5")[0] == "7"
    try:
        _take_flag(["--policy", "pi0", "--action-stride"], "--action-stride", "5")
        raise AssertionError("missing value should have raised")
    except SystemExit:
        pass
    print("  _take_flag handles absent / present / value-missing")


def test_rollout_axis_labels():
    """Rollout plots must say j1..j6/gripper, not a0..a6 — the recorded values are
    absolute joints by then, whatever the checkpoint's action_space string says."""
    import action_record as ar
    from run import _label_rollouts_as_joints
    assert ar.dim_labels(f"delta_joint@{STRIDE}", 7)[0] == "a0"   # before
    _label_rollouts_as_joints()
    assert ar.dim_labels(f"delta_joint@{STRIDE}", 7)[0] == "j1"   # after
    assert ar.dim_labels("delta_eef", 7)[0] == "dx"               # others untouched
    assert ar.dim_labels("joint", 7)[-1] == "gripper"
    print("  rollout axes labelled j1..j6/gripper for delta checkpoints")


# ── speedups ──────────────────────────────────────────────────────────────────

def test_scaled_lr():
    """openpi's pairing is lr 2.5e-5 @ batch 32; sqrt-scale off it."""
    from speedups import scaled_lr, OPENPI_LR
    assert abs(scaled_lr(32) - OPENPI_LR) < 1e-12
    assert abs(scaled_lr(128) - OPENPI_LR * 2) < 1e-12      # sqrt(4) = 2
    assert scaled_lr(64) < OPENPI_LR * 2                    # sqrt, not linear
    print(f"  batch 32 -> {scaled_lr(32):.2e}   64 -> {scaled_lr(64):.2e}   "
          f"128 -> {scaled_lr(128):.2e}")


def test_warmup_cosine_shape():
    """Linear up to peak at warmup, cosine down to the floor, never above peak."""
    from speedups import warmup_cosine
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    total, warm = 1000, 100
    sched = warmup_cosine(opt, total_steps=total, warmup_steps=warm, floor_ratio=0.1)
    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    assert lrs[0] < 0.02, f"warmup must start near zero, got {lrs[0]}"
    assert abs(max(lrs) - 1.0) < 1e-6, f"peak must be the base lr, got {max(lrs)}"
    assert abs(lrs[warm - 1] - 1.0) < 1e-6, "peak should land at the end of warmup"
    assert lrs[warm:] == sorted(lrs[warm:], reverse=True), "must decay monotonically"
    assert 0.09 < lrs[-1] < 0.13, f"should land near the 0.1 floor, got {lrs[-1]}"
    print(f"  warmup {warm} -> peak 1.000 -> floor {lrs[-1]:.3f} over {total} steps")


def test_full_lora_targets_cover_both_towers():
    """The bug this fixes: Gemma names only, so SigLIP got q/k/v and nothing else."""
    from speedups import FULL_LORA_TARGETS
    for n in ("out_proj", "fc1", "fc2"):                    # SigLIP
        assert n in FULL_LORA_TARGETS, n
    for n in ("o_proj", "gate_proj", "up_proj", "down_proj"):  # Gemma
        assert n in FULL_LORA_TARGETS, n
    old = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    # 18 Gemma layers x 7 + 27 SigLIP layers x 3 = 207, the count in the gate log
    assert 18 * len(old & {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                           "up_proj", "down_proj"}) + 27 * 3 == 207
    print("  targets cover Gemma AND SigLIP (old list = the 207-pair gate log)")


def test_init_from_keeps_this_runs_stats():
    """THE trap: a plain load_state_dict would restore the checkpoint's ABSOLUTE
    action stats over the delta ones and silently cancel the whole fix."""
    from speedups import init_from

    class M(torch.nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            self.register_buffer("action_mean", torch.tensor(mean))
            self.register_buffer("action_std", torch.tensor(std))

    old = M([-99.0], [13.0])                                 # absolute checkpoint
    torch.nn.init.constant_(old.lin.weight, 0.5)
    ck = Path(REPO) / "_smoke_dataset" / "_tmp_warmstart.pt"
    torch.save({"model_state": old.state_dict(), "epoch": 9,
                "action_space": "joint"}, ck)
    try:
        new = M([0.0], [2.6])                                # fresh delta stats
        init_from(new, str(ck))
        # float32: 2.6 is not exactly representable, so compare approximately —
        # what matters is it is the delta 2.6 and NOT the checkpoint's 13.0
        assert abs(new.action_mean.item() - 0.0) < 1e-5, "delta stats were clobbered"
        assert abs(new.action_std.item() - 2.6) < 1e-5, "delta stats were clobbered"
        assert abs(new.lin.weight[0, 0].item() - 0.5) < 1e-6, "weights NOT warm-started"
    finally:
        ck.unlink(missing_ok=True)
    print("  init_from: weights loaded, this run's stats preserved")


def test_loader_kwargs():
    from speedups import loader_kwargs
    assert loader_kwargs(8)["persistent_workers"] is True
    assert loader_kwargs(8)["prefetch_factor"] == 4
    assert "persistent_workers" not in loader_kwargs(0)     # invalid at 0 workers
    print("  loader_kwargs valid at 0 and >0 workers")


def test_finetune_flags():
    """'full' is openpi's primary recipe; no mode may freeze the vision tower."""
    from speedups import finetune_flags, FINETUNE_MODES
    full = finetune_flags("full")
    assert full["vlm_lora_rank"] == 0 and not full["train_expert_only"]
    lora = finetune_flags("lora", 16)
    # the bug this guards: train_expert_only=True would make lerobot freeze ALL of
    # paligemma including SigLIP, which openpi's freeze filter never does
    assert lora["vlm_lora_rank"] == 16 and lora["train_expert_only"] is False
    for m in FINETUNE_MODES:
        assert finetune_flags(m)["freeze_vision_encoder"] is False, m
    try:
        finetune_flags("nonsense"); raise AssertionError("should reject")
    except ValueError:
        pass
    print("  full/lora/expert_only flags; no mode freezes the vision tower")


def test_freeze_llm_keep_vision():
    """Gemma frozen, SigLIP + expert + adapters still trainable."""
    from speedups import freeze_llm_keep_vision

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_language_model = torch.nn.Linear(4, 4)
            self.paligemma_vision_tower = torch.nn.Linear(4, 4)
            self.gemma_expert = torch.nn.Linear(4, 4)
            self.paligemma_lora_A = torch.nn.Linear(4, 4)
            # the real lora-mode situation: model.py's `or rank>0` made lerobot
            # freeze EVERYTHING (vision tower included) before this helper runs
            for p in self.parameters():
                p.requires_grad_(False)

    m = freeze_llm_keep_vision(Fake())
    g = dict(m.named_parameters())
    assert not g["paligemma_language_model.weight"].requires_grad, "Gemma not frozen"
    assert g["paligemma_vision_tower.weight"].requires_grad, "SigLIP was frozen!"
    assert g["gemma_expert.weight"].requires_grad, "expert was frozen"
    assert g["paligemma_lora_A.weight"].requires_grad, "adapters were frozen"

    class NoVision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_language_model = torch.nn.Linear(4, 4)

    try:
        freeze_llm_keep_vision(NoVision()); raise AssertionError("should assert")
    except AssertionError as e:
        assert "vision_tower" in str(e), e
    print("  freeze_llm_keep_vision: Gemma frozen, vision/expert/adapters trainable")


def test_pi_family_inference_fixes():
    """int64 attention mask -> torch.where crash on lerobot 0.5.1 (the training
    notebooks patch it; standalone inference had nothing). And a None tokenizer
    must refuse to run, not silently strip language conditioning."""
    from dataset_delta import _fix_pi_family

    class Pi:
        tokenizer = object()
        def _tokenize(self, task, device):
            return torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long)
        def predict(self, obs_state, obs_image=None, task=None): return torch.zeros(7)
        def reset(self): pass

    m = for_inference(Pi(), "delta_joint")
    _, mask = m._tokenize(["x"], "cpu")
    assert mask.dtype == torch.bool, f"mask still {mask.dtype} — torch.where will crash"

    class NoTok(Pi):
        tokenizer = None
    try:
        for_inference(NoTok(), "delta_joint")
        raise AssertionError("None tokenizer must refuse to run")
    except SystemExit:
        pass

    class Act:                                     # no _tokenize: must pass through
        def predict(self, obs_state, obs_image=None, task=None): return torch.zeros(7)
        def reset(self): pass
    for_inference(Act(), "joint")
    print("  mask -> bool, None tokenizer refused, non-language policies untouched")


def test_gate_verdict():
    """The gate must FAIL the measured 30k-run numbers and PASS a working policy."""
    import gate

    def ep(model_err, grip_max, n=100):
        rng = np.random.default_rng(0)
        gt = np.zeros((n, 7)); gt[:, :6] = np.cumsum(rng.normal(0, .1, (n, 6)), 0)
        gt[n // 2:, 6] = 1.0                                     # GT closes
        state = gt.copy(); state[1:, :6] = gt[:-1, :6]           # null err ~0.08 deg
        pred = gt.copy(); pred[:, :6] += model_err; pred[:, 6] = grip_max
        return {"pred": pred, "gt": gt, "state": state}

    # the 30k run: model 1.656 vs null 0.075, gripper max 0.154 -> both criteria fail
    passed, lines = gate.verdict([("ep000", gate.episode_stats(ep(1.656, 0.154)))])
    assert not passed, "gate PASSED the 30k-run numbers!"
    # a policy that beats the null and commits to the grasp -> pass
    passed, _ = gate.verdict([("ep000", gate.episode_stats(ep(0.02, 0.9)))])
    assert passed
    # NaN gt rows (deploy npz has no gt) are excluded, not crashed on
    a = ep(0.02, 0.9); a["gt"][:10] = np.nan
    assert gate.episode_stats(a)["n"] == 90
    print("  gate fails the 30k numbers, passes a working policy, tolerates NaN gt")


def test_loader_uses_file_system_sharing():
    """The bug that killed a pod run at the second epoch: RunPod caps /dev/shm at
    64 MB, 8 workers x prefetch 4 x batch 32 x 2 cameras is ~1.2 GB in flight, and
    the Linux default 'file_descriptor' strategy exhausts it — all workers die with
    'DataLoader worker (pid(s) ...) exited unexpectedly'.

    NOTE macOS only supports file_system, so the transition cannot be observed
    here; this asserts the END STATE, which is what protects the run on Linux.
    """
    from speedups import loader_kwargs, use_file_system_sharing, shm_size_mb
    use_file_system_sharing()
    assert torch.multiprocessing.get_sharing_strategy() == "file_system"
    kw = loader_kwargs(8, 4)
    assert torch.multiprocessing.get_sharing_strategy() == "file_system", \
        "loader_kwargs must set file_system sharing when workers are used"
    assert kw["persistent_workers"] is True and kw["prefetch_factor"] == 4
    # workers=0 must not pass worker-only kwargs (DataLoader rejects them)
    assert "persistent_workers" not in loader_kwargs(0)
    assert "prefetch_factor" not in loader_kwargs(0)
    s = shm_size_mb()
    assert s is None or s > 0
    print("  loader_kwargs forces file_system sharing; workers=0 kwargs are valid")


if __name__ == "__main__":
    for fn in (test_roundtrip, test_stride_picks_every_kth_frame,
               test_stride_extends_the_horizon, test_stats_are_on_the_transformed_actions,
               test_stats_match_getitem, test_absolutise_inverts,
               test_hold_repeats_each_action, test_action_space_string_roundtrips,
               test_for_inference_applies_both, test_for_inference_is_idempotent,
               test_held_actions_are_independent, test_take_flag_edges,
               test_rollout_axis_labels, test_scaled_lr,
               test_warmup_cosine_shape, test_full_lora_targets_cover_both_towers,
               test_init_from_keeps_this_runs_stats, test_loader_kwargs,
               test_finetune_flags, test_freeze_llm_keep_vision,
               test_loader_uses_file_system_sharing,
               test_pi_family_inference_fixes, test_gate_verdict):
        print(f"{fn.__name__}:")
        fn()
    print("\nall action-scale checks passed")
