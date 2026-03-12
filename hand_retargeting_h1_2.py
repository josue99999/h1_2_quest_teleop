import os
import sys
import contextlib
import io
import numpy as np
import mujoco
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TELEOP_DIR = os.path.join(REPO_ROOT, 'teleop')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if TELEOP_DIR not in sys.path:
    sys.path.insert(0, TELEOP_DIR)
_original_cwd = os.getcwd()

def _expand_q6_to_q12(q6: np.ndarray) -> np.ndarray:
    t_yaw, t_pitch, idx, mid, rng, pky = (q6[0], q6[1], q6[2], q6[3], q6[4], q6[5])
    thumb_int = 0.6 * t_pitch
    thumb_dist = 0.35 * t_pitch
    return np.array([t_yaw, t_pitch, thumb_int, thumb_dist, idx, 0.65 * idx, mid, 0.65 * mid, rng, 0.65 * rng, pky, 0.65 * pky], dtype=np.float64)

def build_hand_pose_cache(use_stdout: bool=False):
    from hand_poses_h1_2 import open_hand, closed_fist, pinch, HAND_POSES
    os.chdir(TELEOP_DIR)
    try:
        from inspire_mujoco_retargeting import InspireRetargeting
    finally:
        os.chdir(_original_cwd)
    out = io.StringIO() if not use_stdout else None
    with contextlib.redirect_stdout(out) if out else contextlib.nullcontext():
        retargeter = InspireRetargeting()
    kps_open = open_hand()
    kps_fist = closed_fist()
    kps_pinch = pinch()
    with contextlib.redirect_stdout(out) if out else contextlib.nullcontext():
        left_open_q6 = retargeter.retarget_left(kps_open)
        right_open_q6 = retargeter.retarget_right(kps_open)
        left_fist_q6 = retargeter.retarget_left(kps_fist)
        right_fist_q6 = retargeter.retarget_right(kps_fist)
        left_pinch_q6 = retargeter.retarget_left(kps_pinch)
        right_pinch_q6 = retargeter.retarget_right(kps_pinch)
    return {'left_open': _expand_q6_to_q12(left_open_q6), 'right_open': _expand_q6_to_q12(right_open_q6), 'left_fist': _expand_q6_to_q12(left_fist_q6), 'right_fist': _expand_q6_to_q12(right_fist_q6), 'left_pinch': _expand_q6_to_q12(left_pinch_q6), 'right_pinch': _expand_q6_to_q12(right_pinch_q6)}

def apply_hand_pose_to_mujoco(model, data, left_q12: np.ndarray, right_q12: np.ndarray, hand_left_indices: list, hand_right_indices: list) -> None:
    for i, adr in enumerate(hand_left_indices):
        if i < len(left_q12):
            data.qpos[adr] = left_q12[i]
    for i, adr in enumerate(hand_right_indices):
        if i < len(right_q12):
            data.qpos[adr] = right_q12[i]

def interpolate_hand_poses(hand_pose_cache: dict, t_left: float, t_right: float=None) -> tuple:
    t_left = np.clip(float(t_left), 0.0, 1.0)
    t_right = np.clip(float(t_right), 0.0, 1.0) if t_right is not None else t_left
    left = (1 - t_left) * hand_pose_cache['left_fist'] + t_left * hand_pose_cache['left_open']
    right = (1 - t_right) * hand_pose_cache['right_fist'] + t_right * hand_pose_cache['right_open']
    return (left, right)