# 7. Grasp-Success Gate — requirements (deferred, needs a sensor signal)

One of the deploy-side failures (file 2 §2.4) is the **phantom grasp**: the policy closes the
gripper in mid-air, then — because its only sense of "am I holding something?" is the *binary*
gripper command — believes it succeeded and transitions to the transport phase, retracting with
nothing in hand.

The fix is a **grasp-success gate**: after a close command, *verify a real grasp* before allowing
the arm to proceed; otherwise re-attempt the approach. This is **not implemented** in this branch —
not because of code difficulty, but because **the FR5 gripper exposes no feedback to gate on.**

---

## Why it can't be coded yet

The robot controller (`../so101-fr5-teleop/fr5.py`) is **command-only** for the gripper:

| Available | Method |
|---|---|
| read joint angles | `get_joint_positions()` |
| read TCP/EEF pose | `get_eef_pose()` |
| read joint velocities | `get_joint_velocities()` |
| **command** gripper | `send_gripper(...)` (→ Fairino `MoveGripper`) |
| **read gripper width / force / position** | **— none —** |

There is no `GetGripperStatus` / width / force read in the wrapper (or surfaced from the Fairino
SDK here). So at deploy time the policy (and `deploy.py`) literally cannot tell *closed-on-object*
from *closed-on-air*. Any gate written today would have nothing to check.

---

## What signal is needed (pick one)

| Signal | How | Notes |
|---|---|---|
| **Gripper width / position read** | a Fairino SDK call (if the firmware/gripper supports it) wrapped in `FR5Controller` | best option — directly distinguishes a closed-on-object (jaws stop early) from closed-on-air (jaws fully close). |
| **Gripper force / current** | force-controlled gripper reporting contact force / motor current | a force spike at contact = a real grasp. |
| **Pseudo-tactile (finger angle)** | infer from the force-gripper's own joint angle (no extra hardware) — cf. arXiv:2503.23835 | the cited paper's approach; codeable if the gripper exposes its angle. |
| **Wrist-camera object-in-jaw detector** | a small detector on the wrist frame after the close | no new hardware, but needs a (heuristic or learned) detector and the object visible between the jaws. |
| **Wrist F/T sensor** | a force-torque sensor at the wrist | clean contact signal; needs the sensor. |

---

## Where it plugs in (once a signal exists)

`common/deploy.py`, in the control loop, **between the gripper command and the arm execution**
(after `gripper.update(gripper_cmd)`, before `servo_j` / the delta-EEF IK step):

```python
gripper.update(gripper_cmd)                       # send open/close command
if just_commanded_close:
    grasped = robot.grasp_detected()              # <-- the missing signal (width/force/tactile/vision)
    if not grasped:
        print("[GRASP] mid-air close — re-approaching, not transporting")
        # hold / re-run the approach instead of proceeding to transport
        ... # e.g. lift 5 cm, re-open, retry the reach; do NOT advance the task phase
        continue
# only reaches here (transport) when a grasp is confirmed (or it wasn't a close)
robot.servo_j(joints_cmd)   # or the delta-EEF IK path
```

Until `robot.grasp_detected()` (or equivalent) returns a real signal, the gate is a no-op and is
intentionally left out. Add the wrapper method in `../so101-fr5-teleop/fr5.py` first, then this
gate, then test that a mid-air close triggers a retry rather than a phantom transport.

> Reference: *Disambiguate Gripper State in Grasp-Based Tasks* (arXiv:2503.23835) — the
> open-empty / closed-on-object / closed-on-air ambiguity and a pseudo-tactile fix.
