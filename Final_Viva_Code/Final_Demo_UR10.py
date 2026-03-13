import cv2
import numpy as np
import sys
import rtde_control
import rtde_receive
import time
import requests
import math

# CONFIG
POSE_URL = "http://192.168.227.12:5000/pose"
DEBUG_VIDEO_URL = "http://192.168.227.12:5000/video_feed"   # optional
USE_DEBUG_VIDEO = False
ROBOT_IP = "169.254.106.99"

home_q = [
    1.5738657712936401,
    -1.423939053212301,
    2.6213369369506836,
    -1.1701110045062464,
    1.5673471689224243,
    0.007031369488686323
]

# SERVO TUNING

SERVO_ALPHA = 0.25
SERVO_INTERVAL = 0.12

DEADBAND_X = 0.008
MAX_STEP_X = 0.020
GAIN_X = 0.6

TARGET_DISTANCE_Z = 0.10
DISTANCE_DEADBAND_Z = 0.03
MAX_STEP_Y = 0.020

MOVE_SPEED = 0.20
MOVE_ACCEL = 0.20

POSE_TIMEOUT = 0.8

# SQUARE SCAN MODE TUNING

SCAN_ENABLED = True

SCAN_JOINT_3_INDEX = 3
SCAN_JOINT_4_INDEX = 4

SCAN_STEP_Q3_RAD = math.radians(4.0)
SCAN_STEP_Q4_RAD = math.radians(4.0)

SCAN_MAX_Q3_RAD = math.radians(16.0)
SCAN_MAX_Q4_RAD = math.radians(24.0)

SCAN_INTERVAL = 0.80
SCAN_SPEED = 0.10
SCAN_ACCEL = 0.10
SCAN_START_DELAY = 0.35

# HELPERS

def near_joint_target(current_q, target_q, tol=0.05):
    current_q = np.array(current_q, dtype=float)
    target_q = np.array(target_q, dtype=float)
    return np.max(np.abs(current_q - target_q)) <= tol

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def translate_to_tcp(delta_pose):
    current_pose = rtde_r.getActualTCPPose()
    target_pose = current_pose.copy()

    target_pose[0] = current_pose[0] - delta_pose[0]
    target_pose[1] = current_pose[1] - delta_pose[1]

    target_pose[2] = current_pose[2] + delta_pose[2]
    target_pose[3] = current_pose[3] + delta_pose[3]
    target_pose[4] = current_pose[4] + delta_pose[4]
    target_pose[5] = current_pose[5] + delta_pose[5]

    return target_pose

def build_square_scan_points(step_q3, step_q4, max_q3, max_q4):
    points = [(0.0, 0.0, 0, "center")]
    level = 1

    while True:
        dq3 = level * step_q3
        dq4 = level * step_q4

        if dq3 > max_q3 + 1e-9 or dq4 > max_q4 + 1e-9:
            break

        level_points = [
            (+dq3,  0.0,  level, "+q3"),
            (+dq3, +dq4,  level, "+q3 +q4"),
            ( 0.0, +dq4,  level, "+q4"),
            (-dq3, +dq4,  level, "-q3 +q4"),
            (-dq3,  0.0,  level, "-q3"),
            (-dq3, -dq4,  level, "-q3 -q4"),
            ( 0.0, -dq4,  level, "-q4"),
            (+dq3, -dq4,  level, "+q3 -q4"),
            ( 0.0,  0.0,  level, "center"),
        ]
        points.extend(level_points)
        level += 1

    return points

def get_latest_pose():
    try:
        resp = requests.get(POSE_URL, timeout=0.15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

# UR10 SETUP

print("[INFO] Connecting to UR10...")
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

init_q = rtde_r.getActualQ()
if not near_joint_target(init_q, home_q, tol=0.05):
    print("[INFO] Moving UR10 to home position...")
    rtde_c.moveJ(home_q, 0.1, 0.1)

# Optional debug stream
debug_cap = None
if USE_DEBUG_VIDEO:
    debug_cap = cv2.VideoCapture(DEBUG_VIDEO_URL)

# STATE

move_permission = False
filtered_tvec = None
last_move_time = 0.0
last_seen_time = 0.0

scan_mode = False
scan_center_q = None
scan_points = build_square_scan_points(
    SCAN_STEP_Q3_RAD,
    SCAN_STEP_Q4_RAD,
    SCAN_MAX_Q3_RAD,
    SCAN_MAX_Q4_RAD
)
scan_point_index = 0
last_scan_time = 0.0

last_pose_timestamp = 0.0

print("[INFO] Press 'm' to toggle UR10 movement ON/OFF")
print("[INFO] Press 'h' to send UR10 home")
print("[INFO] Press 'q' to quit")

# MAIN LOOP

try:
    while True:
        now = time.time()
        pose = get_latest_pose()

        marker_found = False
        marker_id = None
        fx = fy = fz = None

        if pose is not None and pose.get("marker_found", False):
            marker_found = True
            marker_id = pose.get("marker_id", None)
            tvec = np.array(pose["tvec"], dtype=float)
            last_pose_timestamp = pose.get("timestamp", now)
            last_seen_time = now

            if filtered_tvec is None:
                filtered_tvec = tvec.copy()
            else:
                filtered_tvec = SERVO_ALPHA * tvec + (1.0 - SERVO_ALPHA) * filtered_tvec

            fx, fy, fz = filtered_tvec

            if scan_mode:
                print("[INFO] Marker found. Exiting square scan mode.")

            scan_mode = False
            scan_center_q = None
            scan_point_index = 0

            if move_permission and (now - last_move_time) > SERVO_INTERVAL:
                move_x = 0.0
                move_y = 0.0

                if abs(fx) > DEADBAND_X:
                    move_x = clamp(GAIN_X * fx, -MAX_STEP_X, MAX_STEP_X)

                z_error = fz - TARGET_DISTANCE_Z

                if z_error > DISTANCE_DEADBAND_Z:
                    move_y = abs(MAX_STEP_Y)
                elif z_error < -DISTANCE_DEADBAND_Z:
                    move_y = -abs(MAX_STEP_Y)
                else:
                    move_y = 0.0

                if abs(move_x) > 1e-6 or abs(move_y) > 1e-6:
                    delta_pose = [move_x, move_y, 0.0, 0.0, 0.0, 0.0]

                    try:
                        target_pose = translate_to_tcp(delta_pose)
                        rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)
                        print(
                            f"[MOVE] ID {marker_id} | "
                            f"fx={fx:.4f}, fz={fz:.4f}, z_err={z_error:.4f} -> "
                            f"dx={move_x:.4f}, dy={move_y:.4f}"
                        )
                    except Exception as e:
                        print("[ERROR] Robot move failed:", e)

                    last_move_time = now
                else:
                    print(
                        f"[INFO] Within deadband | fx={fx:.4f}, fz={fz:.4f}, "
                        f"target_z={TARGET_DISTANCE_Z:.2f}"
                    )

        else:
            pose_age = now - last_seen_time

            if move_permission and SCAN_ENABLED and pose_age > SCAN_START_DELAY:
                if not scan_mode:
                    scan_mode = True
                    scan_center_q = rtde_r.getActualQ()
                    scan_point_index = 0
                    last_scan_time = 0.0
                    filtered_tvec = None
                    print("[INFO] Marker lost. Entering square scan mode using q[3] and q[4].")

                if scan_mode and (now - last_scan_time) > SCAN_INTERVAL:
                    dq3, dq4, level_idx, point_name = scan_points[scan_point_index]

                    target_q = list(scan_center_q)
                    target_q[SCAN_JOINT_3_INDEX] = scan_center_q[SCAN_JOINT_3_INDEX] + dq3
                    target_q[SCAN_JOINT_4_INDEX] = scan_center_q[SCAN_JOINT_4_INDEX] + dq4

                    try:
                        rtde_c.moveJ(target_q, SCAN_SPEED, SCAN_ACCEL)
                        print(
                            f"[SCAN] level={level_idx}, point={point_name}, "
                            f"q3_offset={math.degrees(dq3):.1f} deg, "
                            f"q4_offset={math.degrees(dq4):.1f} deg"
                        )
                    except Exception as e:
                        print("[ERROR] Scan move failed:", e)

                    last_scan_time = now
                    scan_point_index = (scan_point_index + 1) % len(scan_points)

            if pose_age > POSE_TIMEOUT and not scan_mode:
                filtered_tvec = None

        # status display
        status_img = np.zeros((220, 700, 3), dtype=np.uint8)

        cv2.putText(status_img, f"MOVE: {'ON' if move_permission else 'OFF'}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if move_permission else (0,0,255), 2)

        cv2.putText(status_img, f"Marker found: {marker_found}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(status_img, f"Marker ID: {marker_id}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if fx is not None:
            cv2.putText(status_img, f"Filtered X={fx:.3f}  Y={fy:.3f}  Z={fz:.3f}", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(status_img, f"Scan mode: {scan_mode}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        cv2.imshow("UR10 Pose Control Status", status_img)

        if USE_DEBUG_VIDEO and debug_cap is not None:
            ret, dbg = debug_cap.read()
            if ret:
                cv2.imshow("Pi Debug Feed", dbg)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("m"):
            move_permission = not move_permission
            print(f"[INFO] move_permission = {move_permission}")

            if not move_permission:
                scan_mode = False
                scan_center_q = None
                scan_point_index = 0

        elif key == ord("h"):
            print("[INFO] Moving UR10 to home position...")
            try:
                rtde_c.moveJ(home_q, 0.1, 0.1)
                scan_mode = False
                scan_center_q = None
                scan_point_index = 0
                filtered_tvec = None
            except Exception as e:
                print("[ERROR] Home move failed:", e)

finally:
    if debug_cap is not None:
        debug_cap.release()
    cv2.destroyAllWindows()