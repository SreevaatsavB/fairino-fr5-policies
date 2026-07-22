"""
deploy.py — run a trained policy on the FR5 robot.

The checkpoint records which policy produced it, so deploy auto-loads the
matching policies/<policy>/model.py — no per-model deploy script needed.

Usage:
    python common/deploy.py --checkpoint policies/act/checkpoints/best.pt
    python common/deploy.py --checkpoint policies/act/checkpoints/best.pt --steps 150 --no-image
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT   = Path(__file__).resolve().parent.parent
TELEOP_ROOT = REPO_ROOT.parent / "so101-fr5-teleop"
sys.path.insert(0, str(TELEOP_ROOT))

from fr5 import FR5Controller
from config import (
    GRIPPER_INDEX, GRIPPER_TYPE,
    GRIPPER_OPEN_PCT, GRIPPER_CLOSE_PCT,
    GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT, GRIPPER_MAXTIME_MS,
)

sys.path.insert(0, str(REPO_ROOT / "common"))
from action_record import ActionRecorder  # noqa: E402


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_policy_module(policy: str):
    path = REPO_ROOT / "policies" / policy / "model.py"
    if not path.exists():
        raise SystemExit(f"unknown policy '{policy}': {path} not found")
    spec = importlib.util.spec_from_file_location(f"policy_{policy}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POLICY_HZ           = 30
POLICY_PERIOD       = 1.0 / POLICY_HZ
GRIPPER_OPEN_THRESH  = 0.65
GRIPPER_CLOSE_THRESH = 0.35

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── model ─────────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: torch.device, model_overrides: dict | None = None,
                inference_quantize: str | None = None):
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    policy   = ckpt.get("policy", "act")

    # Inference-only overrides (e.g. --te-coeff / --n-action-steps): patch the
    # checkpoint's baked-in model config before build_model. Only safe for knobs
    # that don't change the network architecture / weights.
    if model_overrides:
        for k, v in model_overrides.items():
            old = cfg_dict["model"].get(k, "<unset>")
            cfg_dict["model"][k] = v
            print(f"[deploy] OVERRIDE model.{k}: {old} -> {v}")

    # Post-training quantization for low-VRAM inference (--quantize nf4|int8).
    # Force the BUILD unquantized regardless of what the checkpoint trained with:
    # quantization is applied AFTER load_state_dict below, because a float
    # checkpoint cannot be loaded into params that are already 4-bit. Works even
    # for a checkpoint trained in bf16 — post-hoc NF4 is standard PTQ.
    if inference_quantize:
        prev = cfg_dict["model"].get("quantize", "none")
        cfg_dict["model"]["quantize"] = "none"
        print(f"[deploy] --quantize {inference_quantize}: building unquantized "
              f"(was {prev!r}); quantizing after the weights load")

    # For the VLA policies (pi0/pi05/pi0_fast), skip re-downloading the gated ~6 GB
    # base — the checkpoint's model_state carries every weight.
    from vla_pretrained import strip_pretrained_for_checkpoint  # noqa: E402
    strip_pretrained_for_checkpoint(cfg_dict)

    policy_mod = _load_policy_module(policy)
    model = policy_mod.build_model(cfg_dict, ckpt["stats"], device)
    model.load_state_dict(ckpt["model_state"])

    if inference_quantize:
        if policy not in ("pi0", "pi05", "pi0_fast"):
            raise SystemExit(f"--quantize only applies to the pi-family VLAs, not {policy!r}")
        from vla_pretrained import quantize_for_inference  # noqa: E402
        quantize_for_inference(model, inference_quantize, policy)

    model.eval()

    action_space = ckpt.get("action_space", "joint")
    print(f"loaded {policy} checkpoint  epoch={ckpt['epoch']}  "
          f"val_l1={ckpt['val_l1']:.4f}  action_space={action_space}")
    return model, cfg_dict, action_space, policy


# ── camera ────────────────────────────────────────────────────────────────────

class LiveCamera:
    """Wraps D405Camera to always expose the most recent frame."""

    def __init__(self):
        from camera import D405Camera
        self._cam = D405Camera()

    def start(self):
        self._cam.start()
        self._cam.start_recording()

    def latest_bgr(self) -> np.ndarray | None:
        with self._cam._frames_lock:
            if not self._cam._frames:
                return None
            return self._cam._frames[-1][1].copy()

    def stop(self):
        self._cam.stop_recording()
        self._cam.stop()


def frame_to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return _IMG_TRANSFORM(t).to(device)


# ── gripper ───────────────────────────────────────────────────────────────────

class GripperHandler:
    def __init__(self, robot: FR5Controller):
        self._robot = robot
        self._state = None   # "open" | "closed" | None

    def update(self, gripper_norm: float):
        if gripper_norm >= GRIPPER_OPEN_THRESH and self._state != "open":
            self._send("open", GRIPPER_OPEN_PCT)
        elif gripper_norm <= GRIPPER_CLOSE_THRESH and self._state != "closed":
            self._send("closed", GRIPPER_CLOSE_PCT)

    def _send(self, label: str, pct: int):
        self._robot.stop_servo_mode()
        time.sleep(0.2)

        with self._robot._rpc_lock:
            self._robot._robot.ResetAllError()
        time.sleep(0.05)

        err = self._robot.send_gripper(
            GRIPPER_INDEX, pct,
            GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT,
            GRIPPER_MAXTIME_MS, 1, GRIPPER_TYPE,
        )
        if err == 0:
            self._state = label
            print(f"[GRIPPER] {label.upper()}")
        else:
            print(f"[GRIPPER] MoveGripper error {err}")

        time.sleep(0.1)
        with self._robot._rpc_lock:
            self._robot._robot.RobotEnable(1)
        self._robot.start_servo_mode()


# ── main loop ─────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    overrides = {}
    if args.te_coeff is not None:
        overrides["temporal_ensemble_coeff"] = (
            None if str(args.te_coeff).lower() in ("off", "none", "null")
            else float(args.te_coeff))
    if args.n_action_steps is not None:
        overrides["n_action_steps"] = args.n_action_steps
    model, cfg_dict, action_space, policy = load_policy(
        args.checkpoint, device, overrides, inference_quantize=args.quantize)
    use_image = cfg_dict["dataset"]["use_image"] and not args.no_image

    # Record every predicted action over the rollout (shared schema with the offline
    # eval). Saved on exit — including Ctrl-C — so a rollout is never lost. Plot with
    # tools/plot_actions.py. There is no ground truth on the robot, so this is the
    # prediction-only view of "what the policy commanded over time".
    recorder = None
    if not args.no_log:
        recorder = ActionRecorder(action_space=action_space, policy=policy,
                                  source="deploy",
                                  extra_meta={"checkpoint": str(args.checkpoint),
                                              "task": args.task})

    # Language-conditioned policies (dit_flow, pi0, pi05, pi0_fast) need the task
    # instruction at inference, matching what they were trained on. ACT / Diffusion
    # ignore it. Passed as a batch of one string to model.predict().
    task = [args.task] if args.task else None
    if task:
        print(f"task: {args.task!r}")

    camera  = None
    gripper = None

    try:
        if use_image:
            camera = LiveCamera()
            camera.start()
            print("[CAM] camera ready — waiting for first frame ...")
            for _ in range(30):
                if camera.latest_bgr() is not None:
                    break
                time.sleep(0.1)
            else:
                print("[CAM] no frames received — continuing without image")
                camera.stop()
                camera = None

        with FR5Controller() as robot:
            robot.start_servo_mode()
            gripper = GripperHandler(robot)

            # Activate gripper
            robot.activate_gripper(GRIPPER_INDEX)
            time.sleep(0.5)

            print(f"\nrunning policy for {args.steps} steps at {POLICY_HZ} Hz  "
                  f"({'with' if camera else 'without'} image)\n"
                  "  Ctrl+C to stop early\n")

            model.reset()
            step       = 0
            t_last_log = time.monotonic()
            last_img   = None   # holds last valid frame to avoid None on drops
            last_gripper_cmd = 1.0   # 7th state dim = current gripper opening; assume open at start

            while step < args.steps:
                t0 = time.monotonic()

                # ── observation ──────────────────────────────────────────────
                # state = 6 joint angles (deg) + current gripper opening [0,1] = 7 dims,
                # matching observation.state in the dataset (convert_episodes STATE_COLS).
                # We track the last commanded gripper opening as a proxy for the current
                # one (the FR5 gripper API is command-only, no position feedback). For
                # proprio_mode='none' the model zeros this internally, so it is ignored.
                joints    = robot.get_joint_positions()             # list[float] × 6
                state_vec = joints + [last_gripper_cmd]             # 7-dim
                obs       = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)

                img = last_img
                if camera is not None:
                    bgr = camera.latest_bgr()
                    if bgr is not None:
                        img = frame_to_tensor(bgr, device).unsqueeze(0)
                        last_img = img

                # ── policy ───────────────────────────────────────────────────
                action = model.predict(obs, img, task=task)         # (action_dim,) or (1, action_dim)
                if action.dim() == 2:
                    action = action[0]
                action = action.cpu().numpy()

                gripper_cmd = float(action[6])
                last_gripper_cmd = gripper_cmd   # feeds back into next step's state vector

                # ── execute ───────────────────────────────────────────────────
                gripper.update(gripper_cmd)   # no-op unless threshold crossed
                ik_failed = False
                if action_space == "delta_eef":
                    # action[:6] = TCP pose delta [dx,dy,dz (mm), drx,dry,drz (deg)].
                    # Apply to the CURRENT measured pose, then IK -> joints -> servo.
                    current_eef = np.asarray(robot.get_eef_pose(), dtype=float)
                    target_eef  = (current_eef + np.asarray(action[:6], dtype=float)).tolist()
                    try:
                        joints_cmd = robot.inverse_kin(target_eef)
                        robot.servo_j(joints_cmd)
                    except IOError as e:
                        # unreachable / singular target — hold last pose instead of crashing
                        print(f"[IK] {e} — holding pose")
                        joints_cmd = robot.get_joint_positions()
                        ik_failed = True
                else:  # "joint"
                    joints_cmd = action[:6].tolist()
                    robot.servo_j(joints_cmd)

                if recorder is not None:
                    # executed = the 6 joints actually commanded + the gripper command,
                    # so it lines up dimension-for-dimension with the predicted action.
                    recorder.add(t=step, state=state_vec, pred=action,
                                 executed=list(joints_cmd) + [gripper_cmd],
                                 gripper=gripper_cmd, ik_failed=ik_failed)

                step += 1

                # heartbeat every 2 s
                now = time.monotonic()
                if now - t_last_log >= 2.0:
                    print(f"  step {step}/{args.steps}  "
                          f"joints={[f'{j:.1f}' for j in joints_cmd]}  "
                          f"gripper={gripper_cmd:.2f}")
                    t_last_log = now

                # ── rate limit ────────────────────────────────────────────────
                elapsed = time.monotonic() - t0
                if elapsed < POLICY_PERIOD:
                    time.sleep(POLICY_PERIOD - elapsed)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        if camera is not None:
            camera.stop()
        # save the rollout even if interrupted, so no run is lost
        if recorder is not None and len(recorder) > 0:
            out_dir = Path(args.checkpoint).parent / "rollouts"
            path = recorder.save(out_dir / f"deploy_{_timestamp()}.npz")
            print(f"[LOG] saved {len(recorder)} steps -> {path}")
            print(f"      plot:  python tools/plot_actions.py {path}")

    print("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to best.pt or epoch_XXXX.pt")
    parser.add_argument("--steps",      type=int, default=150,
                        help="number of policy steps to run (default 150 = 5s at 30 Hz)")
    parser.add_argument("--no-image",   action="store_true",
                        help="ignore camera even if checkpoint was trained with images")
    parser.add_argument("--no-log",     action="store_true",
                        help="do not record predicted actions to <ckpt_dir>/rollouts/")
    parser.add_argument("--task", default="pick up the block and place it in the bin",
                        help="language instruction for language-conditioned policies "
                             "(dit_flow / pi0 / pi05 / pi0_fast); ignored by ACT / Diffusion")
    parser.add_argument("--te-coeff", default=None,
                        help="inference-only override: temporal-ensembling coeff for chunk "
                             "policies (e.g. 0.01), or 'off' to disable. Works on existing "
                             "checkpoints — no retraining (dit_flow; ACT reads its own config)")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="inference-only override: receding horizon — execute the first "
                             "k actions of each chunk then re-plan (used when ensembling is "
                             "off; dit_flow)")
    parser.add_argument("--quantize", choices=["nf4", "int8"], default=None,
                        help="pi-family only: post-training quantization of the VLM for "
                             "low-VRAM inference. Works on a checkpoint trained WITHOUT "
                             "quantization (merges any LoRA first, then NF4/int8s the base). "
                             "Needs a Turing-or-newer GPU. pi0 NF4 ~ 3-4 GB total.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
