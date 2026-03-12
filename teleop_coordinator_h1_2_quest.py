from __future__ import annotations
import argparse
import os
import sys
import time
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
from joint_mapping_h1_2 import build_qpos_indices_for_arm, build_hand_qpos_indices, pin_q_to_mjcf_qpos, build_ctrl_indices_for_arm, build_hand_ctrl_indices
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
PELVIS_HEIGHT_WORLD = 1.03
DEFAULT_SCENE = os.path.join(REPO_ROOT, 'H1_2_with_hands', 'scene.xml')
DEFAULT_FPS = 50
HAND_SMOOTH_SPEED = 2.5
CALIBRATION_TRIGGER_THRESHOLD = 0.5
_QUEST_TO_ROS_P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

def _apply_quest_yz_swap(pose_4x4: np.ndarray) -> np.ndarray:
    out = pose_4x4.copy()
    p = np.array([pose_4x4[2, 3], pose_4x4[0, 3], pose_4x4[1, 3]])
    out[:3, 3] = [p[0], -p[1], -p[2]]
    R = pose_4x4[:3, :3]
    out[:3, :3] = _QUEST_TO_ROS_P @ R @ _QUEST_TO_ROS_P.T
    return out

class QuestInput:

    def __init__(self, ip_address: Optional[str]=None, home_left_4x4: Optional[np.ndarray]=None, home_right_4x4: Optional[np.ndarray]=None):
        from meta_quest_teleop.reader import MetaQuestReader
        print(f"[QUEST] Conectando a Quest 3 ({('WiFi: ' + ip_address if ip_address else 'USB')})...")
        self._reader = MetaQuestReader(ip_address=ip_address, run=True)
        print('[QUEST] MetaQuestReader iniciado.')
        self._home_left_4x4 = home_left_4x4 if home_left_4x4 is not None else np.eye(4)
        self._home_right_4x4 = home_right_4x4 if home_right_4x4 is not None else np.eye(4)
        self._left_offset: Optional[np.ndarray] = None
        self._right_offset: Optional[np.ndarray] = None
        self._is_calibrated = False
        self._prev_trigger_pressed = False
        print('[QUEST] Esperando calibración. Presiona TRIGGER DERECHO para calibrar.')

    def update(self) -> Dict:
        reader = self._reader
        left_raw = reader.get_hand_controller_transform_ros('left')
        right_raw = reader.get_hand_controller_transform_ros('right')
        if left_raw is not None:
            left_raw = _apply_quest_yz_swap(left_raw)
        if right_raw is not None:
            right_raw = _apply_quest_yz_swap(right_raw)
        trigger_val = float(reader.get_trigger_value('right'))
        trigger_pressed = trigger_val >= CALIBRATION_TRIGGER_THRESHOLD
        if trigger_pressed and (not self._prev_trigger_pressed) and (left_raw is not None) and (right_raw is not None):
            try:
                self._left_offset = self._home_left_4x4 @ np.linalg.inv(left_raw)
                self._right_offset = self._home_right_4x4 @ np.linalg.inv(right_raw)
                self._is_calibrated = True
                print('[QUEST] ✓ Calibración completada (trigger derecho).')
            except np.linalg.LinAlgError:
                print('[QUEST] ✗ Error de calibración (matrix singular).')
        self._prev_trigger_pressed = trigger_pressed
        left_cal = None
        right_cal = None
        if self._is_calibrated and left_raw is not None and (right_raw is not None):
            left_cal = self._left_offset @ left_raw
            right_cal = self._right_offset @ right_raw
        left_grip = float(reader.get_grip_value('left'))
        right_grip = float(reader.get_grip_value('right'))

        def _btn(name: str) -> bool:
            try:
                return bool(reader.get_button_state(name))
            except Exception:
                return False
        return {'left_4x4': left_cal, 'right_4x4': right_cal, 'left_grip': left_grip, 'right_grip': right_grip, 'is_calibrated': self._is_calibrated, 'button_a': _btn('A'), 'button_b': _btn('B'), 'button_x': _btn('X'), 'button_y': _btn('Y')}

    def close(self):
        try:
            self._reader.stop()
        except Exception:
            pass
        print('[QUEST] MetaQuestReader cerrado.')

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
        print(f'[ARM] Setpoint inicial (FK de L_ee/R_ee):')
        print(f'  Izq EE pos_world: {np.round(self._left_pos_world, 4)}')
        print(f'  Der EE pos_world: {np.round(self._right_pos_world, 4)}')

    def get_home_left_4x4(self) -> np.ndarray:
        T = self._default_left_SE3.copy()
        T[2, 3] += PELVIS_HEIGHT_WORLD
        return T

    def get_home_right_4x4(self) -> np.ndarray:
        T = self._default_right_SE3.copy()
        T[2, 3] += PELVIS_HEIGHT_WORLD
        return T

    def apply_quest_poses(self, left_4x4: np.ndarray, right_4x4: np.ndarray):
        self._left_pos_world[:] = left_4x4[:3, 3]
        self._left_SE3[0, 3] = left_4x4[0, 3]
        self._left_SE3[1, 3] = left_4x4[1, 3]
        self._left_SE3[2, 3] = left_4x4[2, 3] - PELVIS_HEIGHT_WORLD
        self._right_pos_world[:] = right_4x4[:3, 3]
        self._right_SE3[0, 3] = right_4x4[0, 3]
        self._right_SE3[1, 3] = right_4x4[1, 3]
        self._right_SE3[2, 3] = right_4x4[2, 3] - PELVIS_HEIGHT_WORLD
        self._left_SE3[:3, :3] = left_4x4[:3, :3]
        self._right_SE3[:3, :3] = right_4x4[:3, :3]

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
        print('[HAND] Grips analógicos del Quest controlan las manos (suave)')

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
        self._arm_ctrl_indices = build_ctrl_indices_for_arm(self.model)
        self._hand_left_ctrl, self._hand_right_ctrl = build_hand_ctrl_indices(self.model)
        # Initialize all ctrl targets to current qpos (so legs/torso stay put)
        for i in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[jnt_id]]
        # Save free joint addresses and initial pose for pinning
        fj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
        self._fj_qposadr = self.model.jnt_qposadr[fj_id]
        self._fj_dofadr = self.model.jnt_dofadr[fj_id]
        self._fj_init_pos = self.data.qpos[self._fj_qposadr:self._fj_qposadr + 3].copy()
        self._fj_init_quat = self.data.qpos[self._fj_qposadr + 3:self._fj_qposadr + 7].copy()
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
        print(f'[MUJOCO] Listo — nq={self.model.nq}  nu={self.model.nu}  arm_ctrl={len(self._arm_ctrl_indices)}')

    def apply_arm(self, sol_q: np.ndarray):
        for i, ctrl_idx in enumerate(self._arm_ctrl_indices):
            self.data.ctrl[ctrl_idx] = sol_q[i]

    def apply_hands(self, left_q12: Optional[np.ndarray], right_q12: Optional[np.ndarray]):
        if left_q12 is None or right_q12 is None:
            return
        for i, ctrl_idx in enumerate(self._hand_left_ctrl):
            if i < len(left_q12):
                self.data.ctrl[ctrl_idx] = left_q12[i]
        for i, ctrl_idx in enumerate(self._hand_right_ctrl):
            if i < len(right_q12):
                self.data.ctrl[ctrl_idx] = right_q12[i]

    def update_target_viz(self, left_pos: np.ndarray, right_pos: np.ndarray):
        if self._viz_left_mocapid is None or self._viz_left_mocapid < 0:
            return
        self.data.mocap_pos[self._viz_left_mocapid] = left_pos
        self.data.mocap_pos[self._viz_right_mocapid] = right_pos

    def step(self, dt: float):
        n_substeps = max(1, int(np.round(dt / self.model.opt.timestep)))
        qa = self._fj_qposadr
        da = self._fj_dofadr
        for _ in range(n_substeps):
            # Pin free joint before each substep
            self.data.qpos[qa:qa + 3] = self._fj_init_pos
            self.data.qpos[qa + 3:qa + 7] = self._fj_init_quat
            self.data.qvel[da:da + 6] = 0.0
            mujoco.mj_step(self.model, self.data)

class Coordinator:

    def __init__(self, args):
        self.fps = args.fps
        self.dt = 1.0 / self.fps
        self.sim = MuJoCoNode(args.scene)
        self.arm = ArmController()
        self.hand = HandController(smooth_speed=HAND_SMOOTH_SPEED)
        self.quest = QuestInput(ip_address=args.ip_address, home_left_4x4=self.arm.get_home_left_4x4(), home_right_4x4=self.arm.get_home_right_4x4())
        self._prev_button_a = False
        self._prev_button_b = False

    def run(self):
        self._print_help()
        print(f'\n[RUN] Loop a {self.fps} Hz. Viewer abierto (Quest + pink IK).\n')
        print('[RUN] Calentando IK (convergencia inicial)...')
        for _ in range(30):
            sol_q = self.arm.solve()
        if sol_q is not None:
            self.sim.apply_arm(sol_q)
        left_q12, right_q12 = self.hand.update(self.dt)
        self.sim.apply_hands(left_q12, right_q12)
        self.sim.update_target_viz(self.arm.left_pos_world, self.arm.right_pos_world)
        self.sim.step(self.dt)
        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            running = True
            last_status_print = 0.0
            while viewer.is_running() and running:
                t0 = time.monotonic()
                quest_state = self.quest.update()
                btn_b = quest_state['button_b']
                if btn_b and (not self._prev_button_b):
                    running = False
                    self._prev_button_b = btn_b
                    continue
                self._prev_button_b = btn_b
                btn_a = quest_state['button_a']
                if btn_a and (not self._prev_button_a):
                    self.arm.reset()
                self._prev_button_a = btn_a
                if quest_state['is_calibrated']:
                    left_4x4 = quest_state['left_4x4']
                    right_4x4 = quest_state['right_4x4']
                    if left_4x4 is not None and right_4x4 is not None:
                        self.arm.apply_quest_poses(left_4x4, right_4x4)
                sol_q = self.arm.solve()
                if sol_q is not None:
                    self.sim.apply_arm(sol_q)
                target_left = 1.0 - quest_state['left_grip']
                target_right = 1.0 - quest_state['right_grip']
                self.hand.set_targets(target_left, target_right)
                left_q12, right_q12 = self.hand.update(self.dt)
                self.sim.apply_hands(left_q12, right_q12)
                self.sim.update_target_viz(self.arm.left_pos_world, self.arm.right_pos_world)
                self.sim.step(self.dt)
                viewer.sync()
                if t0 - last_status_print >= 2.0:
                    last_status_print = t0
                    cal = '✓' if quest_state['is_calibrated'] else '✗ (presiona trigger R)'
                    print(f"[LOOP] cal={cal}  grips=({quest_state['left_grip']:.2f}, {quest_state['right_grip']:.2f})  izq={np.round(self.arm.left_pos_world, 3)}  der={np.round(self.arm.right_pos_world, 3)}")
                elapsed = time.monotonic() - t0
                sleep_t = max(0.0, self.dt - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        self.quest.close()
        print('\n[EXIT] Coordinador Quest cerrado.')

    @staticmethod
    def _print_help():
        print('\n' + '=' * 64)
        print('  META QUEST 3 → H1_2 TELEOP (pink IK)')
        print('  ─────────────────────────────────────')
        print('  Controladores Quest → posición + orientación de brazos')
        print('  Grip izquierdo      → cerrar/abrir mano izquierda')
        print('  Grip derecho        → cerrar/abrir mano derecha')
        print('  Trigger derecho     → CALIBRAR (adopta pose y presiona)')
        print('  Botón A             → reset brazos')
        print('  Botón B             → salir')
        print('  ─────────────────────────────────────')
        print('  Flujo: conecta Quest → abre viewer → adopta pose → trigger R → mueve mandos')
        print('=' * 64 + '\n')

def parse_args():
    p = argparse.ArgumentParser(description='Coordinador teleop H1_2 — Quest 3 + pink IK')
    p.add_argument('--scene', type=str, default=DEFAULT_SCENE, help='scene.xml H1_2')
    p.add_argument('--fps', type=int, default=DEFAULT_FPS, help='Hz del loop')
    p.add_argument('--ip-address', type=str, default=None, help='IP del Quest 3 (WiFi). Omitir para USB')
    return p.parse_args()
if __name__ == '__main__':
    Coordinator(parse_args()).run()