from __future__ import annotations
import argparse
import os
import select
import sys
import termios
import time
import tty
from typing import Optional, Dict
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
import qpsolvers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
for p in (REPO_ROOT, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
from joint_mapping_h1_2 import build_qpos_indices_for_arm, build_hand_qpos_indices, pin_q_to_mjcf_qpos
H1_2_URDF_PATH = os.path.join(REPO_ROOT, 'assets', 'h1_2', 'h1_2.urdf')
H1_2_ASSETS_DIR = os.path.join(REPO_ROOT, 'assets', 'h1_2')
JOINTS_TO_LOCK = ['left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'torso_joint', 'L_index_proximal_joint', 'L_index_intermediate_joint', 'L_middle_proximal_joint', 'L_middle_intermediate_joint', 'L_pinky_proximal_joint', 'L_pinky_intermediate_joint', 'L_ring_proximal_joint', 'L_ring_intermediate_joint', 'L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint', 'R_index_proximal_joint', 'R_index_intermediate_joint', 'R_middle_proximal_joint', 'R_middle_intermediate_joint', 'R_pinky_proximal_joint', 'R_pinky_intermediate_joint', 'R_ring_proximal_joint', 'R_ring_intermediate_joint', 'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint', 'R_thumb_distal_joint']
LEFT_EE_FRAME = 'L_ee'
RIGHT_EE_FRAME = 'R_ee'
POSITION_COST = 8.0
ORIENTATION_COST = 2.0
EE_LM_DAMPING = 3.0
POSTURE_COST = 0.01
POSTURE_LM_DAMPING = 1.0
IK_DT = 0.05
IK_STEPS_PER_CALL = 3
IK_AMPLIFY = 1.0
POSTURE_JOINT_WEIGHTS = {'shoulder_pitch': 4.0, 'shoulder_roll': 3.0, 'shoulder_yaw': 0.1, 'elbow_pitch': 3.0, 'elbow_roll': 1.0, 'wrist_pitch': 1.0, 'wrist_yaw': 0.1}
DEFAULT_LEFT_POS = np.array([0.282, 0.2095, 1.125])
DEFAULT_RIGHT_POS = np.array([0.282, -0.2095, 1.125])
PELVIS_HEIGHT_WORLD = 1.03
IK_POSITION_SCALE = 1.25
DEFAULT_SCENE = os.path.join(REPO_ROOT, 'H1_2_with_hands', 'scene.xml')
DEFAULT_FPS = 50
KEY_STEP = 0.05
ORI_STEP = np.deg2rad(5)

def _rot_x(a: float) -> np.ndarray:
    c, s = (np.cos(a), np.sin(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(a: float) -> np.ndarray:
    c, s = (np.cos(a), np.sin(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rot_z(a: float) -> np.ndarray:
    c, s = (np.cos(a), np.sin(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
HAND_KEY_RELEASE_THRESHOLD = 0.12
HAND_SMOOTH_SPEED = 2.5

class KeyboardInput:

    def __init__(self):
        import threading
        self._key = None
        self._lock = threading.Lock()
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        tty.setraw(self._fd)
        try:
            while not self._stop:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._key = ch
        finally:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_key(self) -> Optional[str]:
        with self._lock:
            k, self._key = (self._key, None)
            return k

    def close(self):
        self._stop = True
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

class WeightedPostureTask(PostureTask):

    def __init__(self, cost: float, weights: np.ndarray, lm_damping: float=0.0, gain: float=1.0) -> None:
        super().__init__(cost=cost, lm_damping=lm_damping, gain=gain)
        self.weights = weights

    def compute_error(self, configuration):
        return self.weights * super().compute_error(configuration)

    def compute_jacobian(self, configuration):
        return self.weights[:, np.newaxis] * super().compute_jacobian(configuration)

class PinkBodyIKSolver:

    def __init__(self):
        print(f'[PINK IK] Cargando URDF: {H1_2_URDF_PATH}')
        robot_full = pin.RobotWrapper.BuildFromURDF(H1_2_URDF_PATH, H1_2_ASSETS_DIR)
        lock_ids = [j for j in JOINTS_TO_LOCK if robot_full.model.existJointName(j)]
        q_ref = np.zeros(robot_full.model.nq)
        reduced = robot_full.buildReducedRobot(list_of_joints_to_lock=lock_ids, reference_configuration=q_ref)
        self.model = reduced.model
        T_ee_offset = pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]))
        left_wrist_id = self.model.getFrameId('left_wrist_yaw_link')
        left_wrist_frame = self.model.frames[left_wrist_id]
        self.model.addFrame(pin.Frame('L_ee', left_wrist_frame.parentJoint, left_wrist_id, left_wrist_frame.placement * T_ee_offset, pin.FrameType.OP_FRAME))
        right_wrist_id = self.model.getFrameId('right_wrist_yaw_link')
        right_wrist_frame = self.model.frames[right_wrist_id]
        self.model.addFrame(pin.Frame('R_ee', right_wrist_frame.parentJoint, right_wrist_id, right_wrist_frame.placement * T_ee_offset, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        q0 = np.zeros(self.model.nq)
        try:
            if self.model.existJointName('left_shoulder_roll_joint'):
                q0[self.model.joints[self.model.getJointId('left_shoulder_roll_joint')].idx_q] = 0.2
            if self.model.existJointName('right_shoulder_roll_joint'):
                q0[self.model.joints[self.model.getJointId('right_shoulder_roll_joint')].idx_q] = -0.2
        except Exception:
            pass
        self.configuration = pink.Configuration(self.model, self.data, q0)
        self.configuration.model.lowerPositionLimit = self.model.lowerPositionLimit
        self.configuration.model.upperPositionLimit = self.model.upperPositionLimit
        self.task_left = FrameTask(LEFT_EE_FRAME, position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST, lm_damping=EE_LM_DAMPING)
        self.task_right = FrameTask(RIGHT_EE_FRAME, position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST, lm_damping=EE_LM_DAMPING)
        posture_weights = np.ones(self.model.nv)
        for i in range(1, self.model.njoints):
            jname = self.model.names[i]
            for pattern, w in POSTURE_JOINT_WEIGHTS.items():
                if pattern in jname:
                    posture_weights[self.model.joints[i].idx_v] = w
                    break
        self.task_posture = WeightedPostureTask(cost=POSTURE_COST, weights=posture_weights, lm_damping=POSTURE_LM_DAMPING)
        self.all_tasks = [self.task_left, self.task_right, self.task_posture]
        for task in self.all_tasks:
            task.set_target_from_configuration(self.configuration)
        if 'quadprog' in qpsolvers.available_solvers:
            self._solver = 'quadprog'
        else:
            self._solver = qpsolvers.available_solvers[0]
        print(f'[PINK IK] QP solver: {self._solver} — GROOT params (lm_damping={EE_LM_DAMPING})')
        print(f'[PINK IK] Posture weights: {np.round(posture_weights, 1)}')

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        finite = np.isfinite(lower) & np.isfinite(upper)
        q_clipped = q.copy()
        q_clipped[finite] = np.clip(q[finite], lower[finite], upper[finite])
        return q_clipped

    def solve(self, left_SE3: np.ndarray, right_SE3: np.ndarray) -> np.ndarray:
        left_target = pin.SE3(left_SE3[:3, :3], left_SE3[:3, 3])
        right_target = pin.SE3(right_SE3[:3, :3], right_SE3[:3, 3])
        self.task_left.set_target(left_target)
        self.task_right.set_target(right_target)
        for _ in range(IK_STEPS_PER_CALL):
            velocity = solve_ik(self.configuration, self.all_tasks, dt=IK_DT, solver=self._solver)
            self.configuration.q = self._clip_q(self.configuration.q + velocity * IK_DT * IK_AMPLIFY)
            self.configuration.update()
        return self.configuration.q.copy()

    def reset(self):
        q0 = np.zeros(self.model.nq)
        self.configuration = pink.Configuration(self.model, self.data, q0)
        for task in self.all_tasks:
            task.set_target_from_configuration(self.configuration)

class ArmController:

    def __init__(self):
        self.ik = PinkBodyIKSolver()
        left_ee_id = self.ik.model.getFrameId(LEFT_EE_FRAME)
        right_ee_id = self.ik.model.getFrameId(RIGHT_EE_FRAME)
        q0 = self.ik.configuration.q.copy()
        pin.framesForwardKinematics(self.ik.model, self.ik.data, q0)
        self._left_SE3 = np.eye(4)
        self._left_SE3[:3, :3] = self.ik.data.oMf[left_ee_id].rotation.copy()
        self._left_SE3[:3, 3] = self.ik.data.oMf[left_ee_id].translation.copy()
        self._right_SE3 = np.eye(4)
        self._right_SE3[:3, :3] = self.ik.data.oMf[right_ee_id].rotation.copy()
        self._right_SE3[:3, 3] = self.ik.data.oMf[right_ee_id].translation.copy()
        self._left_pos_world = self._left_SE3[:3, 3].copy()
        self._left_pos_world[2] += PELVIS_HEIGHT_WORLD
        self._right_pos_world = self._right_SE3[:3, 3].copy()
        self._right_pos_world[2] += PELVIS_HEIGHT_WORLD
        self._default_left_SE3 = self._left_SE3.copy()
        self._default_right_SE3 = self._right_SE3.copy()
        self._default_left_pos_world = self._left_pos_world.copy()
        self._default_right_pos_world = self._right_pos_world.copy()
        self._current_q = q0.copy()
        print(f'[ARM] Setpoint inicial (FK de L_ee/R_ee, 0 offset):')
        print(f'  Izq EE pos_world: {np.round(self._left_pos_world, 4)}')
        print(f'  Der EE pos_world: {np.round(self._right_pos_world, 4)}')
        print(f'  (pos: ±{KEY_STEP} m/tecla   ori: ±{np.rad2deg(ORI_STEP):.0f}°/tecla)')

    def apply_key(self, key: str) -> bool:
        if key.isupper():
            return self._apply_orientation(key)
        k = key.lower()
        moved = True
        if k == 'w':
            self._left_pos_world[0] += KEY_STEP
        elif k == 's':
            self._left_pos_world[0] -= KEY_STEP
        elif k == 'd':
            self._left_pos_world[1] += KEY_STEP
        elif k == 'a':
            self._left_pos_world[1] -= KEY_STEP
        elif k == 'q':
            self._left_pos_world[2] += KEY_STEP
        elif k == 'e':
            self._left_pos_world[2] -= KEY_STEP
        elif k == 'i':
            self._right_pos_world[0] += KEY_STEP
        elif k == 'k':
            self._right_pos_world[0] -= KEY_STEP
        elif k == 'l':
            self._right_pos_world[1] += KEY_STEP
        elif k == 'j':
            self._right_pos_world[1] -= KEY_STEP
        elif k == 'u':
            self._right_pos_world[2] += KEY_STEP
        elif k == 'o':
            self._right_pos_world[2] -= KEY_STEP
        else:
            moved = False
        if moved:
            self._left_SE3[0, 3] = self._left_pos_world[0]
            self._left_SE3[1, 3] = self._left_pos_world[1]
            self._left_SE3[2, 3] = self._left_pos_world[2] - PELVIS_HEIGHT_WORLD
            self._right_SE3[0, 3] = self._right_pos_world[0]
            self._right_SE3[1, 3] = self._right_pos_world[1]
            self._right_SE3[2, 3] = self._right_pos_world[2] - PELVIS_HEIGHT_WORLD
            print(f'[ARM] izq pos_world={np.round(self._left_pos_world, 3)}  der pos_world={np.round(self._right_pos_world, 3)}')
        return moved

    def _apply_orientation(self, key: str) -> bool:
        moved = True
        if key == 'W':
            self._left_SE3[:3, :3] = _rot_y(ORI_STEP) @ self._left_SE3[:3, :3]
        elif key == 'S':
            self._left_SE3[:3, :3] = _rot_y(-ORI_STEP) @ self._left_SE3[:3, :3]
        elif key == 'A':
            self._left_SE3[:3, :3] = _rot_z(ORI_STEP) @ self._left_SE3[:3, :3]
        elif key == 'D':
            self._left_SE3[:3, :3] = _rot_z(-ORI_STEP) @ self._left_SE3[:3, :3]
        elif key == 'Q':
            self._left_SE3[:3, :3] = _rot_x(ORI_STEP) @ self._left_SE3[:3, :3]
        elif key == 'E':
            self._left_SE3[:3, :3] = _rot_x(-ORI_STEP) @ self._left_SE3[:3, :3]
        elif key == 'I':
            self._right_SE3[:3, :3] = _rot_y(ORI_STEP) @ self._right_SE3[:3, :3]
        elif key == 'K':
            self._right_SE3[:3, :3] = _rot_y(-ORI_STEP) @ self._right_SE3[:3, :3]
        elif key == 'J':
            self._right_SE3[:3, :3] = _rot_z(ORI_STEP) @ self._right_SE3[:3, :3]
        elif key == 'L':
            self._right_SE3[:3, :3] = _rot_z(-ORI_STEP) @ self._right_SE3[:3, :3]
        elif key == 'U':
            self._right_SE3[:3, :3] = _rot_x(ORI_STEP) @ self._right_SE3[:3, :3]
        elif key == 'O':
            self._right_SE3[:3, :3] = _rot_x(-ORI_STEP) @ self._right_SE3[:3, :3]
        else:
            moved = False
        if moved:
            print(f'[ARM] Orientación Shift+{key} aplicada (±{np.rad2deg(ORI_STEP):.0f}°)')
        return moved

    def reset(self):
        self._left_SE3 = self._default_left_SE3.copy()
        self._right_SE3 = self._default_right_SE3.copy()
        self._left_pos_world = self._default_left_pos_world.copy()
        self._right_pos_world = self._default_right_pos_world.copy()
        self.ik.reset()
        print(f'[ARM] Reset → izq={np.round(self._left_pos_world, 3)}  der={np.round(self._right_pos_world, 3)}')

    def solve(self) -> Optional[np.ndarray]:
        try:
            q = self.ik.solve(self._left_SE3, self._right_SE3)
            self._current_q = q.copy()
            return q
        except Exception as exc:
            print(f'[ARM] pink IK falló: {exc}', flush=True)
            return None

    @property
    def left_pos_world(self) -> np.ndarray:
        return self._left_pos_world.copy()

    @property
    def right_pos_world(self) -> np.ndarray:
        return self._right_pos_world.copy()

    @property
    def left_ik_target_world(self) -> np.ndarray:
        return np.array([self._left_SE3[0, 3], self._left_SE3[1, 3], self._left_SE3[2, 3] + PELVIS_HEIGHT_WORLD])

    @property
    def right_ik_target_world(self) -> np.ndarray:
        return np.array([self._right_SE3[0, 3], self._right_SE3[1, 3], self._right_SE3[2, 3] + PELVIS_HEIGHT_WORLD])

class HandController:

    def __init__(self, smooth_speed: float=HAND_SMOOTH_SPEED):
        from hand_retargeting_h1_2 import _expand_q6_to_q12
        from inspire_hand_ranges import open_pose_q6, fist_pose_q6
        self._expand_q6_to_q12 = _expand_q6_to_q12
        self._open_q6 = open_pose_q6()
        self._fist_q6 = fist_pose_q6()
        self._speed = smooth_speed
        self._left_t = 1.0
        self._right_t = 1.0
        self._target_left = 1.0
        self._target_right = 1.0
        print('[HAND] 1 = izq (mantener cerrar, soltar abrir)  2 = der  — movimiento suave')

    def set_targets(self, target_left: float, target_right: float):
        self._target_left = np.clip(float(target_left), 0.0, 1.0)
        self._target_right = np.clip(float(target_right), 0.0, 1.0)

    def update(self, dt: float) -> tuple:
        self._left_t += self._speed * (self._target_left - self._left_t) * dt
        self._right_t += self._speed * (self._target_right - self._right_t) * dt
        self._left_t = np.clip(self._left_t, 0.0, 1.0)
        self._right_t = np.clip(self._right_t, 0.0, 1.0)
        q6_left = (1 - self._left_t) * self._fist_q6 + self._left_t * self._open_q6
        q6_right = (1 - self._right_t) * self._fist_q6 + self._right_t * self._open_q6
        return (self._expand_q6_to_q12(q6_left), self._expand_q6_to_q12(q6_right))

class MuJoCoNode:

    def __init__(self, scene_path: str):
        scene_path = os.path.expanduser(scene_path)
        print(f'[MUJOCO] Cargando: {scene_path}')
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self._arm_indices = build_qpos_indices_for_arm(self.model)
        self._hand_left_indices, self._hand_right_indices = build_hand_qpos_indices(self.model)
        self._viz_left_mocapid = self._viz_right_mocapid = None
        try:
            bid_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_left_viz')
            bid_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_right_viz')
            if bid_l >= 0 and bid_r >= 0:
                self._viz_left_mocapid = self.model.body_mocapid[bid_l]
                self._viz_right_mocapid = self.model.body_mocapid[bid_r]
                print(f'[MUJOCO] Viz mocap IDs: L={self._viz_left_mocapid} R={self._viz_right_mocapid}')
        except Exception:
            pass
        print(f'[MUJOCO] Listo — nq={self.model.nq}  arm={len(self._arm_indices)}')

    def apply_arm(self, sol_q: np.ndarray):
        self.data.qpos[:] = pin_q_to_mjcf_qpos(sol_q, self.model, self.data.qpos, self._arm_indices)

    def apply_hands(self, left_q12: Optional[np.ndarray], right_q12: Optional[np.ndarray]):
        if left_q12 is None or right_q12 is None:
            return
        for i, adr in enumerate(self._hand_left_indices):
            if i < len(left_q12):
                self.data.qpos[adr] = left_q12[i]
        for i, adr in enumerate(self._hand_right_indices):
            if i < len(right_q12):
                self.data.qpos[adr] = right_q12[i]

    def update_target_viz(self, left_pos: np.ndarray, right_pos: np.ndarray):
        if self._viz_left_mocapid is None or self._viz_left_mocapid < 0:
            return
        self.data.mocap_pos[self._viz_left_mocapid] = left_pos
        self.data.mocap_pos[self._viz_right_mocapid] = right_pos

    def step(self):
        mujoco.mj_step(self.model, self.data)

class Coordinator:

    def __init__(self, args):
        self.fps = args.fps
        self.dt = 1.0 / self.fps
        self.sim = MuJoCoNode(args.scene)
        self.arm = ArmController()
        self.hand = HandController(smooth_speed=HAND_SMOOTH_SPEED)
        self.kb = KeyboardInput()
        self._last_key_1_time = 0.0
        self._last_key_2_time = 0.0

    def run(self):
        self._print_help()
        print(f'\n[RUN] Loop a {self.fps} Hz. Viewer abierto (pink IK — pos+rot).\n')
        print('[RUN] Calentando IK (convergencia inicial)...')
        for _ in range(30):
            sol_q = self.arm.solve()
        if sol_q is not None:
            self.sim.apply_arm(sol_q)
        left_q12, right_q12 = self.hand.update(self.dt)
        self.sim.apply_hands(left_q12, right_q12)
        self.sim.update_target_viz(self.arm.left_pos_world, self.arm.right_pos_world)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.sim.data.qvel[:] = 0.0
        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            running = True
            while viewer.is_running() and running:
                t0 = time.monotonic()
                now = t0
                key = self.kb.get_key()
                if key is not None:
                    if key in ('\x1b', 'q'):
                        running = False
                        continue
                    elif key in ('r', 'R'):
                        self.arm.reset()
                    elif key in ('h', 'H'):
                        self._print_help()
                    elif key == '1':
                        self._last_key_1_time = now
                    elif key == '2':
                        self._last_key_2_time = now
                    else:
                        self.arm.apply_key(key)
                target_left = 0.0 if now - self._last_key_1_time <= HAND_KEY_RELEASE_THRESHOLD else 1.0
                target_right = 0.0 if now - self._last_key_2_time <= HAND_KEY_RELEASE_THRESHOLD else 1.0
                self.hand.set_targets(target_left, target_right)
                sol_q = self.arm.solve()
                if sol_q is not None:
                    self.sim.apply_arm(sol_q)
                left_q12, right_q12 = self.hand.update(self.dt)
                self.sim.apply_hands(left_q12, right_q12)
                self.sim.update_target_viz(self.arm.left_pos_world, self.arm.right_pos_world)
                mujoco.mj_forward(self.sim.model, self.sim.data)
                viewer.sync()
                if int(now * 10) % 10 == 0 and (not hasattr(self, '_last_print')) or getattr(self, '_last_print', 0) != int(now):
                    self._last_print = int(now)
                elapsed = time.monotonic() - t0
                sleep_t = max(0.0, self.dt - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        self.kb.close()
        print('\n[EXIT] Coordinador pink cerrado.')

    @staticmethod
    def _print_help():
        print('\n' + '=' * 64)
        print('  PINK IK (pos + orientación)')
        print('  ─── POSICIÓN (minúsculas) ───')
        print('  BRAZO IZQUIERDO       │  BRAZO DERECHO')
        print('  w/s  → +/-X (fwd)    │  i/k  → +/-X')
        print('  a/d  → +/-Y (lat)    │  j/l  → +/-Y')
        print('  q/e  → +/-Z (height) │  u/o  → +/-Z')
        print('  ─── ORIENTACIÓN (Shift / MAYÚSCULAS) ───')
        print('  W/S  → Pitch (±Y)    │  I/K  → Pitch (±Y)')
        print('  A/D  → Yaw   (±Z)    │  J/L  → Yaw   (±Z)')
        print('  Q/E  → Roll  (±X)    │  U/O  → Roll  (±X)')
        print('  ───────────────────────────────────────')
        print('  1  → mano IZQ: mantener=cerrar, soltar=abrir (suave)')
        print('  2  → mano DER: mantener=cerrar, soltar=abrir (suave)')
        print('  R  → reset brazos    H  → ayuda   Esc/q  → salir')
        print('=' * 64 + '\n')

def parse_args():
    p = argparse.ArgumentParser(description='Coordinador teleop H1_2 — pink IK (pos+rot)')
    p.add_argument('--scene', type=str, default=DEFAULT_SCENE, help='scene.xml H1_2')
    p.add_argument('--fps', type=int, default=DEFAULT_FPS, help='Hz del loop')
    return p.parse_args()
if __name__ == '__main__':
    Coordinator(parse_args()).run()