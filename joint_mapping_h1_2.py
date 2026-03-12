import numpy as np
import mujoco
PINOCCHIO_ARM_JOINT_NAMES = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
PIN_TO_MJCF_JOINT_NAME = {'left_elbow_pitch_joint': 'left_elbow_joint', 'left_elbow_roll_joint': 'left_wrist_roll_joint', 'right_elbow_pitch_joint': 'right_elbow_joint', 'right_elbow_roll_joint': 'right_wrist_roll_joint'}

def _get_mjcf_joint_name(pin_name: str) -> str:
    return PIN_TO_MJCF_JOINT_NAME.get(pin_name, pin_name)

def build_qpos_indices_for_arm(mj_model) -> list:
    name_to_adr = {}
    for i in range(mj_model.njnt):
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jname is None:
            continue
        name_to_adr[jname] = mj_model.jnt_qposadr[i]
    indices = []
    for pin_name in PINOCCHIO_ARM_JOINT_NAMES:
        mj_name = _get_mjcf_joint_name(pin_name)
        if mj_name not in name_to_adr:
            raise KeyError(f'Arm joint {mj_name} (from {pin_name}) not found in MuJoCo model. Check that the loaded XML is h1_2_hand.xml (H1_2 with arms).')
        indices.append(name_to_adr[mj_name])
    return indices
HAND_JOINT_NAMES_LEFT = ['L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint', 'L_index_proximal_joint', 'L_index_intermediate_joint', 'L_middle_proximal_joint', 'L_middle_intermediate_joint', 'L_ring_proximal_joint', 'L_ring_intermediate_joint', 'L_pinky_proximal_joint', 'L_pinky_intermediate_joint']
HAND_JOINT_NAMES_RIGHT = ['R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint', 'R_thumb_distal_joint', 'R_index_proximal_joint', 'R_index_intermediate_joint', 'R_middle_proximal_joint', 'R_middle_intermediate_joint', 'R_ring_proximal_joint', 'R_ring_intermediate_joint', 'R_pinky_proximal_joint', 'R_pinky_intermediate_joint']

def build_hand_qpos_indices(mj_model) -> tuple:
    name_to_adr = {}
    for i in range(mj_model.njnt):
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jname is not None:
            name_to_adr[jname] = mj_model.jnt_qposadr[i]

    def get_indices(names):
        return [name_to_adr[n] for n in names if n in name_to_adr]
    return (get_indices(HAND_JOINT_NAMES_LEFT), get_indices(HAND_JOINT_NAMES_RIGHT))

def pin_q_to_mjcf_qpos(sol_q: np.ndarray, mj_model, current_qpos: np.ndarray, arm_qpos_indices: list) -> np.ndarray:
    qpos = np.array(current_qpos, dtype=np.float64, copy=True)
    for i, qpos_adr in enumerate(arm_qpos_indices):
        qpos[qpos_adr] = sol_q[i]
    return qpos