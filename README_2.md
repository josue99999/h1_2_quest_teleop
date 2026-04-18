# H1_2 Meta Quest 3 Teleoperation

Real-time arm and hand teleoperation of the Unitree H1_2 humanoid robot in MuJoCo,
controlled with a Meta Quest 3 VR headset.

![Teleoperation demo](teleop_h1.gif)

---

## What This Project Does

The operator puts on a Meta Quest 3 headset, moves their hands in the air, and the robot's arms follow in real time — matching both position and orientation (full SE(3) tracking) at 50 Hz.

At the same time, squeezing the analog grip on each controller opens and closes the robot's robotic hands.

Everything runs inside a MuJoCo physics simulation. No real robot is required.

This is the **upper-body control layer** that pairs with the lower-body RL locomotion policy from [VELOCITY_RL](https://github.com/josue99999/VELOCITY_RL), building toward a full whole-body teleoperation system.

---

## How It Works — Overview

```
Operator moves hands
        │
        ▼
Meta Quest 3  ──────  sends hand position + orientation + grip values
        │
        ▼
Coordinate remapping  ── converts Quest axis convention to robot frame
        │
        ▼
Calibration  ────────── aligns operator's "zero pose" with robot's rest pose
        │
        ▼
Differential IK solver  ─  computes which arm joints to move and by how much
        │
        ▼
MuJoCo simulation  ─────  robot arm follows the command at 50 Hz
```

---

## Arm Control — How the IK Works

Moving a robot arm from a hand position is not trivial. The robot has many joints, and there are infinite joint combinations that reach the same point. The system uses **Differential Inverse Kinematics** — at every timestep it solves a small optimization problem to find the best joint movement.

The optimization balances three goals at once:

| Goal | What it does |
|---|---|
| Match left hand position and orientation | Follow the operator's left controller |
| Match right hand position and orientation | Follow the operator's right controller |
| Stay close to a natural resting posture | Avoid weird or twisted arm configurations |

Position is weighted higher than orientation, so the robot prioritizes reaching the right location before matching the exact wrist angle.

This runs **3 micro-iterations per control cycle** at 50 Hz, giving smooth and stable tracking without lag.

The solver only moves the arm joints — legs, torso, and fingers are kept fixed during IK to keep the problem simple and fast.

---

## Hand Control — Grip to Finger Movement

Hand control works differently from arm control. It is simpler by design:

- Squeeze the grip → hand closes into a fist
- Release the grip → hand opens fully
- Intermediate squeeze → proportional open/close position

The system interpolates smoothly between a fully open and fully closed configuration across all 12 finger joints per hand. A small filter prevents abrupt jumps when the grip value changes quickly.

The 12 joints are driven from 6 representative angles, with the distal (tip) joints always set to 65% of the proximal (base) joints — matching how the physical Inspire robotic hand mechanics actually work.

---

## Calibration

When the teleoperation starts, the robot and the operator's hands are not aligned — the Quest measures positions in its own coordinate space, not the robot's.

**To calibrate:**
1. Hold your arms in the robot's default rest position
2. Press the right index trigger once

The system records the difference between your current Quest pose and the robot's known home pose. From that moment, all movements are interpreted as **relative to that reference**, so moving your hand 10 cm to the left moves the robot's hand 10 cm to the left.

---

## Coordinate Frame Remapping

The Meta Quest uses a different axis convention than the robot controller expects (different X/Y/Z orientation and handedness). Before any computation, every incoming hand pose is remapped so that moving your hand forward moves the robot's hand forward — not sideways or backward.

This is applied automatically and transparently every cycle.

---

## Base Pinning

The H1_2 model in MuJoCo has a free-floating base joint, which means without any control the robot would fall over due to gravity. Since this project only controls the arms and hands (the legs are not driven here), the base is **pinned in place** at every physics step — position, orientation, and velocity are reset to the starting pose before each simulation step.

This keeps the torso stable so the operator can focus entirely on arm and hand teleoperation.

---

## Key Numbers

| Setting | Value | What it means |
|---|---|---|
| Control loop | 50 Hz | How often arm commands are updated |
| IK iterations per cycle | 3 | Micro-steps per control tick for smoother convergence |
| Position weight | 8.0 | How hard the solver works to match hand position |
| Orientation weight | 2.0 | How hard the solver works to match wrist angle |
| Posture regularization | 0.01 | How strongly it avoids awkward arm poses |
| Distal/proximal ratio | 0.65 | Fixed finger joint coupling for hand closing |

---

## Top 3 Hardest Problems Solved

### Problem 1 — IK Instability Near Arm Singularities

When the arm is fully extended or in certain configurations, the math behind IK breaks down — the solver produces extremely large joint velocity commands that cause the simulation to explode. This is called a kinematic singularity and is a well-known problem in robot control. I solved it by tuning the Levenberg-Marquardt damping parameter, which adds a small penalty that prevents the solver from producing extreme outputs near these problem configurations. I also added hard joint velocity clipping as a second safety layer, and adjusted the robot's home pose so the operator's natural working range stays away from singular configurations. The result is smooth, stable tracking across the full practical arm workspace.

### Problem 2 — Quest-to-Robot Coordinate Frame Alignment

The Meta Quest headset measures hand positions in its own coordinate system, which has a different axis layout and handedness than the robot's control frame. This means a raw mapping makes the robot's arm move in completely wrong directions — left feels like up, forward feels like sideways. Fixing this required testing each individual axis of movement separately, observing what the robot did, and deducing the exact permutation and sign flip needed for correct alignment. I also built a standalone 3D visualization tool (`validate_quest_h1_2.py`) that shows the remapped poses in real time so the alignment can be verified before launching the full simulation. The result is intuitive one-to-one correspondence between the operator's hand motion and the robot's arm motion.

### Problem 3 — Natural Hand Closing Across 12 Finger Joints

The Inspire robotic hands have 12 joints per hand, but controlling all of them independently from a single analog grip value is complex — a naive uniform approach makes fingers close at unnatural rates and produces unrealistic grasps. Real underactuated robotic hands have mechanical coupling between proximal and distal finger joints that must be respected. I implemented a 6-DOF representative control space and derived fixed expansion ratios (distal = 0.65 × proximal) from the hand's mechanical design, applied with a smooth first-order filter to avoid velocity spikes. The result is natural-looking grasp motion that closely matches how the physical Inspire hand closes, making the simulation directly applicable to real hardware deployment.

---

## Usage

```bash
# USB connection
python teleop_coordinator_h1_2_quest.py

# WiFi connection
python teleop_coordinator_h1_2_quest.py --ip-address 192.168.x.x

# No Quest? Use keyboard instead
python teleop_coordinator_h1_2_pink.py
```

### Controls

| Input | Action |
|---|---|
| Move controllers | Move robot arms (position + orientation) |
| Analog grips | Open / close hands |
| Right index trigger (click) | Calibrate — hold rest pose first |
| A button | Reset arms to home position |
| B button | Exit |

---

## Related Repositories

- [VELOCITY_RL](https://github.com/josue99999/VELOCITY_RL) — lower-limb RL locomotion (legs)
- [video2robot_h1_2](https://github.com/josue99999/video2robot_h1_2) — human video to robot motion data

---

## License

MIT
