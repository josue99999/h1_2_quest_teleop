import numpy as np
FINGER_MIN, FINGER_MAX = (0.0, 1.7)
THUMB_PITCH_MIN, THUMB_PITCH_MAX = (0.0, 0.5)
THUMB_YAW_MIN, THUMB_YAW_MAX = (-0.1, 1.3)

def open_pose_q6() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

def fist_pose_q6() -> np.ndarray:
    return np.array([0.0, 0.5, 1.7, 1.7, 1.7, 1.7], dtype=np.float64)