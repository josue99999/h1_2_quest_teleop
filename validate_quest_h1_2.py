import argparse
import os
import sys
import time
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for p in (REPO_ROOT, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from meta_quest_teleop.reader import MetaQuestReader
except ImportError:
    print('[ERROR] meta_quest_teleop no instalado.')
    sys.exit(1)
try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
    from gear_sonic.utils.teleop.readers.quest_reader import quest_controllers_to_vr_3pt_pose
except ImportError as e:
    print(f'[ERROR] Dependencias de gear_sonic (VR3PtPoseVisualizer) faltantes: {e}')
    sys.exit(1)
from teleop_coordinator_h1_2_quest import PinkBodyIKSolver, _apply_quest_yz_swap

def main():
    parser = argparse.ArgumentParser(description='Validar tracking Meta Quest 3 para H1_2')
    parser.add_argument('--ip-address', type=str, default=None, help='Quest IP (WiFi)')
    parser.add_argument('--hz', type=float, default=50.0, help='Frecuencia (Hz)')
    parser.add_argument('--calibrated', action='store_true', help='Usar FK del H1_2 como referencia de calibración')
    parser.add_argument('--duration', type=float, default=0.0, help='Duración en segundos (0=infinito)')
    args = parser.parse_args()
    home_left_4x4 = None
    home_right_4x4 = None
    if args.calibrated:
        print('[validador] Inicializando modelo H1_2 para FK de calibración...')
        ik = PinkBodyIKSolver()
        left_ee_id = ik.model.getFrameId('L_ee')
        right_ee_id = ik.model.getFrameId('R_ee')
        home_left_4x4 = np.eye(4)
        home_left_4x4[:3, :3] = ik.data.oMf[left_ee_id].rotation.copy()
        home_left_4x4[:3, 3] = ik.data.oMf[left_ee_id].translation.copy()
        home_left_4x4[2, 3] += 1.03
        home_right_4x4 = np.eye(4)
        home_right_4x4[:3, :3] = ik.data.oMf[right_ee_id].rotation.copy()
        home_right_4x4[:3, 3] = ik.data.oMf[right_ee_id].translation.copy()
        home_right_4x4[2, 3] += 1.03
        print('[validador] Poses por defecto extraídas de Pinocchio (H1_2).')
    print('=' * 60)
    print('Quest Tracking Validation (H1_2)')
    print('=' * 60)
    if args.calibrated:
        print('  Modo: CALIBRADO (H1_2 FK como referencia)')
        print('  1. Adopta la pose por defecto del robot (brazos bajados hacia el frente)')
        print('  2. Pulsa el TRIGGER DERECHO para calibrar el offset')
    else:
        print('  Modo: RAW (sin calibración, solo poses crudas)')
    print('=' * 60)
    reader = MetaQuestReader(ip_address=args.ip_address, run=True)
    dt = 1.0 / max(1.0, args.hz)
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=False, robot_model=None)
    visualizer.create_realtime_plotter(interactive=True)
    left_offset = None
    right_offset = None
    is_calibrated = False
    prev_trigger_pressed = False
    last_status = 0.0
    frame_count = 0
    fps_ema = 0.0
    last_stamp = time.monotonic()
    start_time = time.time()
    try:
        while visualizer.is_open:
            t_now = time.monotonic()
            device_dt = t_now - last_stamp
            if device_dt > 0:
                inst_fps = 1.0 / device_dt
                fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps
            last_stamp = t_now
            left_raw = reader.get_hand_controller_transform_ros('left')
            right_raw = reader.get_hand_controller_transform_ros('right')
            if left_raw is not None:
                left_raw = _apply_quest_yz_swap(left_raw)
            if right_raw is not None:
                right_raw = _apply_quest_yz_swap(right_raw)
            trigger = float(reader.get_trigger_value('right'))
            trigger_pressed = trigger >= 0.5
            if args.calibrated and trigger_pressed and (not prev_trigger_pressed):
                if left_raw is not None and right_raw is not None and (home_left_4x4 is not None) and (home_right_4x4 is not None):
                    try:
                        left_offset = home_left_4x4 @ np.linalg.inv(left_raw)
                        right_offset = home_right_4x4 @ np.linalg.inv(right_raw)
                        is_calibrated = True
                        print('\n[validador] Calibración completada (offset L/R calculado).')
                    except np.linalg.LinAlgError:
                        pass
            prev_trigger_pressed = trigger_pressed
            if args.calibrated and is_calibrated and (left_offset is not None) and (right_offset is not None) and (left_raw is not None) and (right_raw is not None):
                left_pose = left_offset @ left_raw
                right_pose = right_offset @ right_raw
                vr_3pt_pose = quest_controllers_to_vr_3pt_pose(left_pose, right_pose)
            else:
                vr_3pt_pose = quest_controllers_to_vr_3pt_pose(left_raw, right_raw)
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()
            frame_count += 1
            now = time.time()
            if now - last_status >= 2.0:
                status = 'L:OK' if left_raw is not None else 'L:--'
                status += ' R:OK' if right_raw is not None else ' R:--'
                calib_str = ' [CALIB]' if args.calibrated and is_calibrated else ''
                print(f'\r[validador] {status} | {fps_ema:.1f} Hz | frames={frame_count}{calib_str}   ', end='', flush=True)
                last_status = now
            if args.duration > 0 and now - start_time >= args.duration:
                print(f'\n[validador] Duración {args.duration}s alcanzada.')
                break
            time.sleep(dt)
    except KeyboardInterrupt:
        print('\n[validador] Interrumpido.')
    finally:
        reader.stop()
        visualizer.close()
if __name__ == '__main__':
    main()