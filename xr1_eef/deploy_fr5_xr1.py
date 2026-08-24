"""
deploy_fr5_xr1.py — drive the FR5 from an XR-1 inference server. Runs on the robot
PC, inside the XR-1 `mibot` env with the teleop repo on PYTHONPATH.

    # on the GPU box (or the same PC):  bash scripts/deploy.sh <posttrained_dir> 1 1
    python xr1_eef/deploy_fr5_xr1.py --task "Pick up each blue block and put it in the brown tray." \
        --host 127.0.0.1 --steps 900 --execute 15

Loop, at 30 Hz:
  FR5 state  ->  XR-1 state dict   (joints deg->rad, eef mm/deg -> m/rotm, gripper 0/1)
  2 cameras  ->  PIL               (ego = scene D435i, wrist_left = wrist D405)
  server     ->  (30, 60) chunk    (their Client, subclassed for a 2-view prompt)
  recover_action -> absolute targets in the base frame (their io.py, verbatim)
  rotm -> Fairino RPY (extrinsic XYZ)  ->  GetInverseKin  ->  ServoJ
  execute the first --execute entries, then re-plan (receding horizon, like delta_joint)

NOT yet run on hardware. Every primitive it calls (get_joint_positions, get_eef_pose,
inverse_kin, servo_j, the gripper handler) is the one common/deploy.py already uses.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "common"), str(REPO.parent / "so101-fr5-teleop"), str(Path(__file__).resolve().parent)]

from fr5_to_xr1 import rotm_to_rpy_deg, rpy_deg_to_rotm  # noqa: E402  (same convention as training)


class TwoViewClient:
    """XR-1's runtime Client with the prompt built for our two cameras. Everything
    else (socket framing, state packing, recover_action) is theirs, unchanged."""

    def __init__(self, host, port):
        from mibot.server.runtime.client import Client
        self._c = Client(host, port)

    def __call__(self, robot_state, ego, wrist, instruction):
        from mibot.utils.io import compose_state, recover_action, resize_image, split_action
        import torch
        c = self._c
        ego, wrist = (resize_image(im, factor=32, max_pixels=160000) for im in (ego, wrist))
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
            {"type": "image", "image": ego},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": wrist},
            {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
        ]}, {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]}]
        payload = c.processor.apply_chat_template([messages], tokenize=True, return_dict=True,
                                                  return_tensors="pt", padding=True,
                                                  images_kwargs={"do_resize": False})
        payload["state"] = torch.from_numpy(compose_state(
            left_gripper=robot_state["left_gripper_pos"], left_joint=robot_state["left_arm_joint"],
            right_gripper=np.zeros(1, np.float32), right_joint=np.zeros(6, np.float32)))[None]
        c._send(payload)
        action = np.asarray(c._recv().numpy(), dtype=np.float32)
        action = action[0] if action.shape == (1, 30, 60) else action
        rs = dict(robot_state, right_ee_pos=np.zeros(3), right_ee_rotm=np.eye(3), right_gripper_pos=np.zeros(1))
        return recover_action(action, rs), split_action(action)


def fr5_state(robot, last_gripper_cmd):
    """XR-1 robot_state from the FR5: joints rad, ee m + rotm, gripper (commanded, 0/1)."""
    joints_deg = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    eef = np.asarray(robot.get_eef_pose(), dtype=np.float64)          # mm, deg
    return {
        "left_arm_joint": np.deg2rad(joints_deg).astype(np.float32),
        "left_gripper_pos": np.array([last_gripper_cmd], np.float32),
        "left_ee_pos": (eef[:3] / 1000.0).astype(np.float32),
        "left_ee_rotm": rpy_deg_to_rotm(eef[None, 3:6])[0].astype(np.float32),
    }


def targets_to_fairino(targets, k):
    """k-th absolute target -> Fairino [x,y,z mm, rx,ry,rz deg]."""
    pos_mm = targets["left_ee_pos"][k] * 1000.0
    rpy = rotm_to_rpy_deg(targets["left_ee_rotm"][k][None])[0]
    return [*pos_mm.tolist(), *rpy.tolist()], float(targets["left_gripper_pos"][k][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=10086)
    ap.add_argument("--task", required=True, help="one of the 9 dataset sentences")
    ap.add_argument("--steps", type=int, default=900, help="control steps at 30 Hz (900 = 30 s)")
    ap.add_argument("--execute", type=int, default=15,
                    help="chunk entries executed before re-planning (30 = the full 1 s chunk)")
    ap.add_argument("--no-robot", action="store_true", help="query the server with a fake state, print, exit")
    args = ap.parse_args()

    client = TwoViewClient(args.host, args.port)

    if args.no_robot:
        st = {"left_arm_joint": np.zeros(6, np.float32), "left_gripper_pos": np.zeros(1, np.float32),
              "left_ee_pos": np.array([0, 0.3, 0.38], np.float32), "left_ee_rotm": np.eye(3, dtype=np.float32)}
        blank = Image.new("RGB", (640, 480))
        targets, parts = client(st, blank, blank, args.task)
        print("server OK. entry-0 Δpos (tool frame, m):", parts["left_ee_pos"][0].round(4),
              " Δaa:", parts["left_ee_aa"][0].round(4), " grip:", parts["left_gripper"][0].round(3))
        print("entry-0 target (fairino mm/deg):", np.round(targets_to_fairino(targets, 0)[0], 2))
        return

    import cv2
    from deploy import GripperHandler, LiveCamera, POLICY_PERIOD, GRIPPER_CLOSE_THRESH  # common/deploy.py
    from fr5 import FR5Controller
    import camera as cam_mod
    # the robot PC's camera.py has RealSenseCamera (used by the 07-28 launcher for the
    # D435i); older copies only have D405Camera, which takes a serial too
    SCENE_SERIAL = "420122071835"                                   # D435i, verified 2026-07-26
    if hasattr(cam_mod, "RealSenseCamera"):
        scene = cam_mod.RealSenseCamera(serial=SCENE_SERIAL, name="scene_cam", enable_depth=False)
    else:
        scene = cam_mod.D405Camera(serial=SCENE_SERIAL)

    wrist_cam = LiveCamera(); wrist_cam.start()
    scene.start(); scene.start_recording()
    to_pil = lambda bgr: Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    with FR5Controller() as robot:
        robot.start_servo_mode()
        gripper = GripperHandler(robot); gripper._state = "open"
        last_grip = 0.0
        step = 0
        while step < args.steps:
            state = fr5_state(robot, last_grip)
            with scene._frames_lock:
                scene_bgr = scene._frames[-1][1].copy() if scene._frames else None
            wrist_bgr = wrist_cam.latest_bgr()
            if scene_bgr is None or wrist_bgr is None:
                time.sleep(POLICY_PERIOD); continue
            t_plan = time.perf_counter()
            targets, _ = client(state, to_pil(scene_bgr), to_pil(wrist_bgr), args.task)
            print(f"step {step:4d}  plan {1000*(time.perf_counter()-t_plan):.0f} ms")
            for k in range(min(args.execute, 30)):
                t0 = time.perf_counter()
                pose, grip = targets_to_fairino(targets, k)
                gripper.update(1.0 if grip >= GRIPPER_CLOSE_THRESH else 0.0); last_grip = float(grip >= GRIPPER_CLOSE_THRESH)
                try:
                    robot.servo_j(robot.inverse_kin(pose))
                except IOError as e:
                    print(f"  [IK] {e} — holding")
                step += 1
                dt = time.perf_counter() - t0
                if dt < POLICY_PERIOD:
                    time.sleep(POLICY_PERIOD - dt)
    scene.stop_recording(); scene.stop(); wrist_cam.stop()


if __name__ == "__main__":
    main()
