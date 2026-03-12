# H1_2 Meta Quest 3 Teleoperation

This repository provides a standalone MuJoCo teleoperation environment for the Unitree H1_2 robot using Meta Quest 3 tracking. It employs a whole-body differential Inverse Kinematics (IK) solver named `PinkBodyIKSolver` (based on [pink](https://github.com/stephane-caron/pink) and Pinocchio).

## Capabilities

- **VR Tracking**: Uses Meta Quest 3 controllers for full SE(3) end-effector tracking (position and orientation) via `meta_quest_teleop`.
- **Differential IK**: Real-time QP-based inverse kinematics mapping SE(3) tracking to robot joint states.
- **Hand Control**: Maps the Meta Quest analog grips to the opening and closing of the Inspire robotic hands (smooth interpolation).
- **Validation Tools**: Includes 3D visualization and command-line scripts to validate tracking and calibration data independently.

## Requirements and Dependencies

This project is standalone but relies on several external Python libraries. We recommend using a virtual environment (e.g. `venv`, `conda`, `uv`).

```bash
# Core Simulation and Math
pip install numpy scipy mujoco

# Kinematics and Inverse Kinematics
pip install pin pink-qpsolvers qpsolvers[quadprog]

# Meta Quest 3 Teleoperation
pip install 'meta_quest_teleop @ git+https://github.com/BrikHMP18/meta_quest_teleop.git@07bc15437f767c3517138367b6b3e3910b388c76'

# 3D Validation Visualizer
pip install pyvista
```

> **Note**: An Android Debug Bridge (ADB) connection to your Meta Quest 3 device is required (either via USB or WiFi). Ensure Developer Mode is activated on your headset.

## Usage Guide

### 1. Connecting to the Quest

Turn on your Meta Quest 3, connect it via USB (or know its IP address for WiFi) and ensure the ADB connection is active:
```bash
adb devices
```

### 2. Validation Tools (Optional)

Before running the full physics simulation, you can validate the raw poses coming from the Quest or test the calibration reference frame.

**Console Raw Verification:**
```bash
python test_quest_h1_2.py
# If using WiFi: python test_quest_h1_2.py --ip-address 192.168.x.x
```

**3D Calibration Verification (PyVista):**
```bash
python validate_quest_h1_2.py --calibrated
```
- Stand in the robot's default pose (arms resting downwards).
- Press the **Right Trigger** to set the calibration offset. The 3D plot should now mirror your hand movements faithfully.

### 3. Launching the MuJoCo Coordinator

Launch the physics simulation and control loop:

```bash
# USB Connection
python teleop_coordinator_h1_2_quest.py

# WiFi Connection
python teleop_coordinator_h1_2_quest.py --ip-address 192.168.x.x
```

#### Controls inside MuJoCo:

- **Move Hands**: Drive the VR controllers to move the robot's end-effectors fully in SE(3).
- **Analog Grips (L/R)**: Squeeze the triggers on the side of the handles to smoothly close the robot's hands. Release to open.
- **Right Index Trigger (Click)**: Calibrate the offset. Adopt the robot's rest pose and quickly press the index trigger completely.
- **A Button**: Reset both arms to their default stable position.
- **B Button**: Exit the application safely.

## Keyboard Fallback

If you don't have the Quest available, you can still test the IK solver using keyboard inputs for Position (XYZ) and Orientation (RPY):
```bash
python teleop_coordinator_h1_2_pink.py
```
*(Press `H` inside the MuJoCo window to view the keybindings).*
