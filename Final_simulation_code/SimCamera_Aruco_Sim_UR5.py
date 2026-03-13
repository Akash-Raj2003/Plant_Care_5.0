import time
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# connect to CoppeliaSim and get sim object.
client = RemoteAPIClient()
sim = client.getObject('sim')

# find the UR5 joints using candidate paths
def get_ur5_joints():
    candidate_sets = [
        [
            '/UR5/joint',
            '/UR5/joint/joint',
            '/UR5/joint/joint/joint',
            '/UR5/joint/joint/joint/joint',
            '/UR5/joint/joint/joint/joint/joint',
            '/UR5/joint/joint/joint/joint/joint/joint',
        ],
        [
            '/OmniPlatform/body/UR5/joint',
            '/OmniPlatform/body/UR5/joint/joint',
            '/OmniPlatform/body/UR5/joint/joint/joint',
            '/OmniPlatform/body/UR5/joint/joint/joint/joint',
            '/OmniPlatform/body/UR5/joint/joint/joint/joint/joint',
            '/OmniPlatform/body/UR5/joint/joint/joint/joint/joint/joint',
        ]
    ]

    last_error = None
    for paths in candidate_sets:
        try:
            joints = [sim.getObject(p) for p in paths]
            return joints
        except Exception as e:
            last_error = e

    raise RuntimeError(f'Could not find UR5 joints. Last error: {last_error}')

UR5_JOINTS = get_ur5_joints()
J1, J2, J3, J4, J5, J6 = UR5_JOINTS

# vision sensor retrieval using candidate paths
def get_vision_sensor():
    candidate_paths = [
        '/ArucoVisionSensor',
        '/OmniPlatform/body/UR5/ArucoVisionSensor',
        '/OmniPlatform/body/UR5/joint/joint/joint/joint/joint/joint/connection/ArucoVisionSensor',
        '/OmniPlatform/body/UR5/connection/ArucoVisionSensor',
    ]

    last_error = None
    for p in candidate_paths:
        try:
            h = sim.getObject(p)
            return h, p
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f'Could not find Vision Sensor. Rename it to "ArucoVisionSensor". Last error: {last_error}'
    )

vision_sensor, vision_path = get_vision_sensor()
print('Using vision sensor:', vision_path)

# aruco setup for marker detection
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# camera helper to get frames from the vision sensor
def get_sim_frame(sensor_handle):
    img, width, height = sim.getVisionSensorCharImage(sensor_handle)
    frame = np.frombuffer(img, dtype=np.uint8).reshape(height, width, 3)
    frame = cv2.flip(frame, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

#utility function to clamp values within limits
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def set_joint_targets(q):
    sim.setJointTargetPosition(J1, q[0])
    sim.setJointTargetPosition(J2, q[1])
    sim.setJointTargetPosition(J3, q[2])
    sim.setJointTargetPosition(J4, q[3])
    sim.setJointTargetPosition(J5, q[4])
    sim.setJointTargetPosition(J6, q[5])

# tuning parameters for control
SIGN_J1 = +1.0
SIGN_J2 = -1.0

Kp_pan = 0.8
Kp_tilt = 0.8
control_step = 0.04

TARGET_EX = 0.0
TARGET_EY = 0.55

LOCK_TOL_X = 0.10
LOCK_TOL_Y = 0.12

SEARCH_SWEEP_AMPLITUDE = 0.6
SEARCH_SWEEP_SPEED = 0.45
MAX_LOST_FRAMES = 15

alpha = 0.30

# setting joint limits for clamping
J1_MIN, J1_MAX = -3.0, 3.0
J2_MIN, J2_MAX = -2.2, 2.2
J3_MIN, J3_MAX = -2.5, 2.5
J4_MIN, J4_MAX = -3.0, 3.0
J5_MIN, J5_MAX = -3.0, 3.0
J6_MIN, J6_MAX = -3.0, 3.0

# home and search configurations
home_q = [
    1.5739017724990845,
    -0.5636547719137,
    1.5957118034362793,
    2.4929574171649378,
    1.5673471689224243,
    0.007079306524246931
]

search_q = home_q.copy()
search_q[1] = home_q[1] - 0.25
search_q[2] = home_q[2] + 0.20

# main function
sim.startSimulation()
time.sleep(0.5)

state = 'SEARCH'
lost_count = 0
ex_f = 0.0
ey_f = 0.0
err_x = 0.0
err_y = 0.0

locked_q1 = None
locked_q2 = None

try:
    set_joint_targets(home_q)
    time.sleep(1.5)

    set_joint_targets(search_q)
    time.sleep(1.0)

    start_time = time.time()

    while True:
        t = time.time() - start_time
        frame = get_sim_frame(vision_sensor)
        h, w = frame.shape[:2]

        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)
        marker_found = ids is not None and len(corners) > 0

        if state == 'SEARCH':
            sweep_j1 = home_q[0] + SEARCH_SWEEP_AMPLITUDE * math.sin(SEARCH_SWEEP_SPEED * t)

            sim.setJointTargetPosition(J1, clamp(sweep_j1, J1_MIN, J1_MAX))
            sim.setJointTargetPosition(J2, clamp(search_q[1], J2_MIN, J2_MAX))
            sim.setJointTargetPosition(J3, clamp(search_q[2], J3_MIN, J3_MAX))
            sim.setJointTargetPosition(J4, clamp(search_q[3], J4_MIN, J4_MAX))
            sim.setJointTargetPosition(J5, clamp(search_q[4], J5_MIN, J5_MAX))
            sim.setJointTargetPosition(J6, clamp(search_q[5], J6_MIN, J6_MAX))

            if marker_found:
                state = 'TRACK'
                lost_count = 0

        elif state == 'TRACK':
            if marker_found:
                lost_count = 0

                c = corners[0].reshape(4, 2).astype(np.float32)
                mx = float(np.mean(c[:, 0]))
                my = float(np.mean(c[:, 1]))

                ex = (mx - (w / 2)) / (w / 2)
                ey = (my - (h / 2)) / (h / 2)

                ex_f = (1 - alpha) * ex_f + alpha * ex
                ey_f = (1 - alpha) * ey_f + alpha * ey

                err_x = ex_f - TARGET_EX
                err_y = ey_f - TARGET_EY

                q1 = sim.getJointPosition(J1)
                q2 = sim.getJointPosition(J2)

                q1_cmd = q1 + control_step * (SIGN_J1 * Kp_pan * err_x)
                q2_cmd = q2 + control_step * (SIGN_J2 * Kp_tilt * err_y)

                q1_cmd = clamp(q1_cmd, J1_MIN, J1_MAX)
                q2_cmd = clamp(q2_cmd, J2_MIN, J2_MAX)

                sim.setJointTargetPosition(J1, q1_cmd)
                sim.setJointTargetPosition(J2, q2_cmd)

                sim.setJointTargetPosition(J3, clamp(search_q[2], J3_MIN, J3_MAX))
                sim.setJointTargetPosition(J4, clamp(search_q[3], J4_MIN, J4_MAX))
                sim.setJointTargetPosition(J5, clamp(search_q[4], J5_MIN, J5_MAX))
                sim.setJointTargetPosition(J6, clamp(search_q[5], J6_MIN, J6_MAX))

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                if abs(err_x) < LOCK_TOL_X and abs(err_y) < LOCK_TOL_Y:
                    locked_q1 = q1_cmd
                    locked_q2 = q2_cmd
                    state = 'LOCKED'

            else:
                lost_count += 1
                if lost_count > MAX_LOST_FRAMES:
                    state = 'SEARCH'
                    lost_count = 0

        elif state == 'LOCKED':
            if locked_q1 is not None:
                sim.setJointTargetPosition(J1, locked_q1)
            if locked_q2 is not None:
                sim.setJointTargetPosition(J2, locked_q2)

            sim.setJointTargetPosition(J3, clamp(search_q[2], J3_MIN, J3_MAX))
            sim.setJointTargetPosition(J4, clamp(search_q[3], J4_MIN, J4_MAX))
            sim.setJointTargetPosition(J5, clamp(search_q[4], J5_MIN, J5_MAX))
            sim.setJointTargetPosition(J6, clamp(search_q[5], J6_MIN, J6_MAX))

            if marker_found:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.putText(
            frame,
            f'STATE: {state}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        target_y_px = int((TARGET_EY * (h / 2)) + (h / 2))
        cv2.line(frame, (0, target_y_px), (w, target_y_px), (255, 255, 255), 1)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

        if state == 'TRACK':
            cv2.putText(
                frame,
                f'ex={ex_f:+.2f} ey={ey_f:+.2f}',
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f'err_x={err_x:+.2f} err_y={err_y:+.2f}',
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        if state == 'LOCKED':
            cv2.putText(
                frame,
                'Locked at bottom-center',
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.imshow('UR5 ArUco Bottom Lock', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        time.sleep(1 / 30)

finally:
    sim.stopSimulation()
    cv2.destroyAllWindows()
    print('Simulation stopped.')