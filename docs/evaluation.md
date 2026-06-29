# Action logging & evaluation layer

A small, cross-policy layer for **recording predicted actions over time** and
**comparing them to ground truth**. It answers two questions:

1. *On the robot* — what did the policy actually command over the rollout? (`deploy.py`)
2. *On the test set* — does the policy reproduce the demonstrated actions? (`eval_on_test.py`)

It is a **horizontal layer**: one shared record format + one shared plotter, used by both
the online (robot) and offline (dataset) paths, and working for every lerobot-style policy
(`act`, `diffusion`, `dit_flow`, `pi0`, `pi05`, `pi0_fast`) through the common
`model.predict()` / `model.reset()` interface. (Octo has its own inference pipeline and is
not covered here.)

## The pieces

| File | Role |
|---|---|
| `common/action_record.py` | `ActionRecorder` — one on-disk schema (`.npz` + `.json` sidecar) for "actions over time", shared by deploy and eval. Ground truth is optional (present offline, absent on the robot). |
| `tools/plot_actions.py` | Plot any record: **one subplot per action dimension**, over time. Auto-detects GT-vs-prediction (offline) vs prediction-only (robot). |
| `common/deploy.py` | Records every step of a robot rollout, saved to `<ckpt_dir>/rollouts/` (even on Ctrl-C). |
| `experiments/eval_on_test.py` | Replays held-out **test** episodes through the policy and saves GT-vs-prediction records to `<ckpt_dir>/eval/`. |

Dimension labels come from the checkpoint's `action_space`:
`joint` → `[j1…j6, gripper]`, `delta_eef` → `[dx, dy, dz, drx, dry, drz, gripper]`.

## 1. Robot rollout (online)

Logging is on by default — every `deploy.py` run drops a record:

```bash
python common/deploy.py --checkpoint policies/act/checkpoints/best.pt --steps 300
# -> policies/act/checkpoints/rollouts/deploy_<timestamp>.npz  (+ .json)
python tools/plot_actions.py policies/act/checkpoints/rollouts/deploy_<timestamp>.npz
```

The record holds, per step: the input `state`, the `pred`icted action, the `executed`
joint command + gripper, the gripper command, and an `ik_failed` flag (delta-EEF runs).
There is no ground truth on the robot, so the plot is prediction-only. Disable with
`--no-log`.

## 2. Test-set evaluation (offline, GT vs prediction)

Runs the policy over the **held-out test episodes** — the same split
`FR5Dataset.episode_split(val_frac, seed)` that `train.py` never trained on — and plots the
predicted action against the demonstrated (ground-truth) action, per dimension, over time:

```bash
python experiments/eval_on_test.py --ckpt policies/act/checkpoints/best.pt
# default: first 2 held-out episodes -> <ckpt_dir>/eval/ep<NNN>.npz + .png
python experiments/eval_on_test.py --ckpt best.pt --episodes 3            # first 3 test eps
python experiments/eval_on_test.py --ckpt best.pt --episodes 0 5 9        # specific eps
python experiments/eval_on_test.py --ckpt best.pt --max-steps 300         # cap per episode
```

Each episode produces a per-dimension GT-vs-prediction plot (with per-dim **MAE** in each
subplot title) and a printed MAE table. Lower MAE = the policy reproduces the demonstration
more faithfully.

### What this measures (and what it does not)

This evaluation is **open-loop / teacher-forced**: at every step the policy sees the
*recorded* observation from the dataset, not an observation produced by its own previous
action. So it measures **single-step prediction accuracy** — "given this exact situation,
does the policy output the demonstrated action?" — not closed-loop behaviour.

It will **not** reveal compounding error / covariate shift (the policy drifting into states
it never saw — see `docs/failure_analysis/02`), because the observations never drift. A
policy can have low offline MAE here and still fail on the robot. Use this to check action
reproduction and to compare checkpoints; use a real rollout (and the
`experiments/shortcut_probe.py` vision-vs-state probe) for the rest.

To reproduce the robot path exactly, `eval_on_test.py` calls `model.reset()` at the start of
each episode and then `predict()` sequentially in time order — the same way `deploy.py`
drives the policy (including ACT's temporal ensembling).
