"""
run.py — train / deploy / eval with the action-scale fixes. See README.md.

    python delta_joint/run.py train  --policy pi0 --action-stride 5
    python delta_joint/run.py deploy --checkpoint <best.pt> --task "..."
    python delta_joint/run.py eval   --ckpt <best.pt>

Own flags (train only; everything else passes straight through to common/train.py):

    --action-stride K   chunk samples every Kth frame          (default 5)
    --no-delta          absolute joint targets, stride only    (default: delta on)

Deploy and eval take NO scale flags — they read what to do out of the checkpoint's
action_space string, so a checkpoint can never be run at the wrong speed or with
the state left off.

Nothing in common/, policies/ or experiments/ changes. Training swaps the dataset
class; inference wraps model.predict, so deploy.py's existing "joint" branch is
untouched.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "common"), str(REPO / "experiments"),
                str(Path(__file__).resolve().parent)]

from dataset_delta import (DeltaJointDataset, for_inference,  # noqa: E402
                           parse_action_space)


def _take_flag(argv, name, default=None, is_bool=False):
    """Pull our own flag out of argv so the rest reaches the wrapped main()."""
    if name not in argv:
        return default, argv
    i = argv.index(name)
    if is_bool:
        return True, argv[:i] + argv[i + 1:]
    if i + 1 >= len(argv):
        raise SystemExit(f"{name} needs a value")
    return argv[i + 1], argv[:i] + argv[i + 2:]


def cmd_train(argv):
    stride, argv = _take_flag(argv, "--action-stride", "5")
    no_delta, argv = _take_flag(argv, "--no-delta", False, is_bool=True)
    stride, delta = int(stride), not no_delta

    import train as train_mod
    base = train_mod.FR5Dataset

    def make(*a, **kw):
        return DeltaJointDataset(*a, action_stride=stride, delta=delta, **kw)

    make.episode_split = base.episode_split      # train.py calls it on the class
    train_mod.FR5Dataset = make

    print(f"[delta] training: delta={delta}  action_stride={stride}  "
          f"(chunk spans {stride}x longer in time)")
    sys.argv = [sys.argv[0]] + argv
    train_mod.main()


def _label_rollouts_as_joints():
    """ActionRecorder looks axis labels up by the raw action_space string, so
    'delta_joint@5' falls back to a0..a6 and every rollout plot becomes unreadable.
    By the time the recorder sees them the values ARE absolute joints (absolutise
    already ran), so map them onto the 'joint' labels."""
    import action_record as ar
    original = ar.dim_labels
    ar.dim_labels = lambda space, dim: original(
        parse_action_space(space)[0].replace("delta_joint", "joint"), dim)


def cmd_deploy(argv):
    import deploy
    _label_rollouts_as_joints()
    original = deploy.load_policy

    def load_policy_delta(*a, **kw):
        model, cfg, action_space, policy = original(*a, **kw)
        base, _ = parse_action_space(action_space)
        if base not in ("delta_joint", "joint"):
            raise SystemExit(f"action_space={action_space!r} is not handled here; "
                             f"use common/deploy.py")
        for_inference(model, action_space)
        return model, cfg, action_space, policy   # deploy's else-branch = joint

    deploy.load_policy = load_policy_delta
    sys.argv = [sys.argv[0]] + argv
    deploy.main()


def cmd_eval(argv):
    import eval_on_test as ev
    original = ev.eval_episode

    # eval_episode already receives the checkpoint's action_space string, so hook
    # there rather than second-guessing it. Called once per episode; the wrappers
    # are idempotent, and the state-added-twice bug that would otherwise cause is
    # covered by test_for_inference_is_idempotent.
    def eval_episode(model, ds, action_space, policy, device):
        for_inference(model, action_space)
        return original(model, ds, action_space, policy, device)

    ev.eval_episode = eval_episode
    print("[delta] eval against ABSOLUTE ground truth (predictions un-scaled)")
    sys.argv = [sys.argv[0]] + argv
    ev.main()


def main():
    cmds = {"train": cmd_train, "deploy": cmd_deploy, "eval": cmd_eval}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(f"usage: run.py {{{'|'.join(cmds)}}} [args...]\n{__doc__}")
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
