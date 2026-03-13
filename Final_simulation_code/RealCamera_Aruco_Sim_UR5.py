import time
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

#coppelia sim remote api client setup and UR5 joint handle retrieval. 
# The code first initializes the RemoteAPIClient and retrieves the sim object. 
# It then defines a function get_ur5_joints that tries two different sets of 
# possible paths to find the UR5 joint handles in the CoppeliaSim scene. 
# If it successfully finds the joints, it returns their handles along with the paths 
# used. If it fails to find the joints using both sets of paths, it raises a 
# RuntimeError with the last error encountered.
client = RemoteAPIClient()
sim = client.getObject("sim")


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
            return joints, paths
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not find UR5 joints by path. Last error: {last_error}")

UR5_JOINTS, UR5_PATHS = get_ur5_joints()

J1 = UR5_JOINTS[0]  # base
J2 = UR5_JOINTS[1]  # shoulder
J3 = UR5_JOINTS[2]  # elbow
J4 = UR5_JOINTS[3]  # wrist1
J5 = UR5_JOINTS[4]  # wrist2
J6 = UR5_JOINTS[5]  # wrist3

print("Using UR5 joint paths:")
for p, h in zip(UR5_PATHS, UR5_JOINTS):
    print(f"  {p} -> {h}")

#opencv aruco setup. 
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

MARKER_SIZE_M = 0.05  # Real marker size in meters

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# helper functions for pose estimation and control. 
def approx_intrinsics(w, h):
    # Rough intrinsics for basic pose estimation.
    # For better pose quality, calibrate your camera.
    f = float(w)
    K = np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist = np.zeros((5,), dtype=np.float32)
    return K, dist

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rvec_to_yaw_pitch_roll(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        yaw   = 0.0

    return float(yaw), float(pitch), float(roll)

# gains for proportional control. 
Kp_pan  = 0.9
Kp_tilt = 0.9

Kp_yaw_orient   = 0.25
Kp_pitch_orient = 0.25

Kp_roll_orient = 0.20
USE_WRIST_ROLL = False

# safety limits for joint commands
J1_MIN, J1_MAX = -3.0, 3.0
J2_MIN, J2_MAX = -2.2, 2.2
J4_MIN, J4_MAX = -3.0, 3.0

# smoothed pose estimates
yaw_f = 0.0
pitch_f = 0.0
roll_f = 0.0
cx_f = 0.0
cy_f = 0.0

# initial pose for the robot
def set_initial_pose():
    try:
        sim.setJointTargetPosition(J1, 0.0)
        sim.setJointTargetPosition(J2, -0.6)
        sim.setJointTargetPosition(J3, 0.9)
        sim.setJointTargetPosition(J4, 0.0)
        sim.setJointTargetPosition(J5, 0.0)
        sim.setJointTargetPosition(J6, 0.0)
        time.sleep(1.0)
    except Exception as e:
        print("Could not set initial pose:", e)

# main function 
sim.startSimulation()
time.sleep(0.5)

try:
    set_initial_pose()
    print("Running. Press 'q' in the webcam window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        K, dist = approx_intrinsics(w, h)

        corners, ids, _ = DETECTOR.detectMarkers(frame)

        if ids is not None and len(corners) > 0:
            c = corners[0].reshape(4, 2).astype(np.float32)

            mx = float(np.mean(c[:, 0]))
            my = float(np.mean(c[:, 1]))

            # normalized error
            ex = (mx - (w / 2)) / (w / 2)
            ey = (my - (h / 2)) / (h / 2)

            s = MARKER_SIZE_M
            obj_pts = np.array([
                [-s/2,  s/2, 0],
                [ s/2,  s/2, 0],
                [ s/2, -s/2, 0],
                [-s/2, -s/2, 0],
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                c,
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            alpha = 0.25
            cx_f = (1 - alpha) * cx_f + alpha * ex
            cy_f = (1 - alpha) * cy_f + alpha * ey

            if success:
                yaw, pitch, roll = rvec_to_yaw_pitch_roll(rvec)
                yaw_f   = (1 - alpha) * yaw_f   + alpha * yaw
                pitch_f = (1 - alpha) * pitch_f + alpha * pitch
                roll_f  = (1 - alpha) * roll_f  + alpha * roll

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Read current joint positions
            q1 = sim.getJointPosition(J1)
            q2 = sim.getJointPosition(J2)
            q4 = sim.getJointPosition(J4)

            # Primary objective: center marker
            dq1_center = +Kp_pan * cx_f
            dq2_center = -Kp_tilt * cy_f

            # Secondary objective: reduce marker orientation error
            dq1_orient = -Kp_yaw_orient * yaw_f if success else 0.0
            dq2_orient = +Kp_pitch_orient * pitch_f if success else 0.0

            q1_cmd = q1 + 0.02 * (dq1_center + dq1_orient)
            q2_cmd = q2 + 0.02 * (dq2_center + dq2_orient)

            q1_cmd = clamp(q1_cmd, J1_MIN, J1_MAX)
            q2_cmd = clamp(q2_cmd, J2_MIN, J2_MAX)

            sim.setJointTargetPosition(J1, q1_cmd)
            sim.setJointTargetPosition(J2, q2_cmd)

            if USE_WRIST_ROLL and success:
                q4_cmd = q4 + 0.02 * (-Kp_roll_orient * roll_f)
                q4_cmd = clamp(q4_cmd, J4_MIN, J4_MAX)
                sim.setJointTargetPosition(J4, q4_cmd)

            cv2.putText(
                frame,
                f"ex={cx_f:+.2f} ey={cy_f:+.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"yaw={yaw_f:+.2f} pitch={pitch_f:+.2f} roll={roll_f:+.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.imshow("UR5 Align to ArUco", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        time.sleep(1 / 30)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    sim.stopSimulation()
    cap.release()
    cv2.destroyAllWindows()
    print("Simulation stopped.")