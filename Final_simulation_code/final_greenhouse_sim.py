from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import threading
import queue
import math
import numpy as np
import cv2

# connect to CoppeliaSim remote API
client = RemoteAPIClient()
sim = client.getObject('sim')

# find omni directional robot joints and waypoints in the scene
joint_paths = [
    '/OmniPlatform/link[0]/regularRotation',
    '/OmniPlatform/link[1]/regularRotation',
    '/OmniPlatform/link[2]/regularRotation',
    '/OmniPlatform/link[3]/regularRotation',
]
joints = [sim.getObject(p) for p in joint_paths]
robot = sim.getObject('/OmniPlatform')

waypoint_paths = [
    '/wp1', '/wp2', '/wp3', '/wp4', '/wp5',
    '/wp6', '/wp7', '/wp8', '/wp9', '/wp10',
]
waypoints = [sim.getObject(p) for p in waypoint_paths]

# find UR5 joints with robust error handling to support different scene hierarchies
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
            found_joints = [sim.getObject(p) for p in paths]
            return found_joints
        except Exception as e:
            last_error = e

    raise RuntimeError(f'Could not find UR5 joints. Last error: {last_error}')

UR5_JOINTS = get_ur5_joints()
J1, J2, J3, J4, J5, J6 = UR5_JOINTS

# find vision sensor with robust error handling to support different scene hierarchies
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

# set up ArUco marker detection
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# settings and tuning parameters
v = 100 * 2.398795 * math.pi / 180.0
pos_tolerance = 0.10
control_dt = 0.05
settle_time = 0.2

# tuning parameters for UR5 ArUco tracking control
SIGN_J1 = +1.0
SIGN_J2 = -1.0

Kp_pan = 0.8
Kp_tilt = 0.8
arm_control_step = 0.04

SEARCH_SWEEP_AMPLITUDE = 1.1
SEARCH_SWEEP_SPEED = 0.85
MAX_LOST_FRAMES = 20
ARUCO_LOCK_TIMEOUT = 15.0

# marker is considered "locked" if visible for these many frames
VISIBLE_HOLD_FRAMES = 4

alpha = 0.30

# set joint limits for the UR5 arm to prevent it from hitting the greenhouse structure during sweeping/searching
J1_MIN, J1_MAX = -3.0, 3.0
J2_MIN, J2_MAX = -2.2, 2.2
J3_MIN, J3_MAX = -2.5, 2.5
J4_MIN, J4_MAX = -3.0, 3.0
J5_MIN, J5_MAX = -3.0, 3.0
J6_MIN, J6_MAX = -3.0, 3.0

# home and search poses for the UR5 arm. Search pose is more downward-looking to help find the marker on the plant.
home_q = [
    1.5739017724990845,
    -0.5636547719137,
    1.5957118034362793,
    2.4929574171649378,
    1.5673471689224243,
    0.007079306524246931
]

# More downward-looking search pose than before
search_q = home_q.copy()
search_q[1] = home_q[1] - 0.55
search_q[2] = home_q[2] + 0.35

# controlled plants and their associated waypoints
LEAF_COLOR_NAME = "LEAVES"
YELLOW = [1.00, 1.00, 0.59]
GREEN = [0.598, 1.00, 0.59]

plant_shape_map = {
    "p0":  "/indoorPlant[0]/visible",
    "p1":  "/indoorPlant[1]/visible",
    "p2":  "/indoorPlant[2]/visible",
    "p3":  "/indoorPlant[3]/visible",
    "p4":  "/indoorPlant[4]/visible",
    "p5":  "/indoorPlant[5]/visible",
    "p6":  "/indoorPlant[6]/visible",
    "p7":  "/indoorPlant[7]/visible",
    "p8":  "/indoorPlant[8]/visible",
    "p9":  "/indoorPlant[9]/visible",
    "p10": "/indoorPlant[10]/visible",
    "p11": "/indoorPlant[11]/visible",
    "p12": "/indoorPlant[12]/visible",
    "p13": "/indoorPlant[13]/visible",
}

plant_waypoint_map = {
    "p0":  [1],
    "p1":  [2],
    "p2":  [3],
    "p3":  [4],
    "p4":  [5],
    "p5":  [6],
    "p6":  [7],
    "p7":  [8],
    "p8":  [9],
    "p9":  [10],
    "p10": [1, 10],
    "p11": [2, 9],
    "p12": [3, 8],
    "p13": [4, 7],
}

plant_is_yellow = {name: False for name in plant_shape_map}

waypoint_plant_map = {i: [] for i in range(1, 11)}
for plant_name, wp_list in plant_waypoint_map.items():
    for wp in wp_list:
        waypoint_plant_map[wp].append(plant_name)

# shared state and synchronization primitives
command_queue = queue.Queue()
stop_requested = False
current_wp = None
pending_plant_commands = set()

# helper function to get vision sensor image as a proper OpenCV BGR frame
def get_sim_frame(sensor_handle):
    img, width, height = sim.getVisionSensorCharImage(sensor_handle)
    frame = np.frombuffer(img, dtype=np.uint8).reshape(height, width, 3)
    frame = cv2.flip(frame, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

#utility function to clamp values within specified limits
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def set_joint_targets(q):
    sim.setJointTargetPosition(J1, q[0])
    sim.setJointTargetPosition(J2, q[1])
    sim.setJointTargetPosition(J3, q[2])
    sim.setJointTargetPosition(J4, q[3])
    sim.setJointTargetPosition(J5, q[4])
    sim.setJointTargetPosition(J6, q[5])

# wheel control functions for the omni-directional robot. The move_pos_x, move_neg_x, move_pos_y, move_neg_y functions set the wheel velocities to move in the corresponding direction. The set_wheels function is a helper that directly sets the velocities of all four wheels, and the stop function sets all wheel velocities to zero.
def set_wheels(v0, v1, v2, v3):
    sim.setJointTargetVelocity(joints[0], v0)
    sim.setJointTargetVelocity(joints[1], v1)
    sim.setJointTargetVelocity(joints[2], v2)
    sim.setJointTargetVelocity(joints[3], v3)

def stop():
    set_wheels(0.0, 0.0, 0.0, 0.0)

def move_pos_x():
    set_wheels(-v, -v,  v,  v)

def move_neg_x():
    set_wheels( v,  v, -v, -v)

def move_pos_y():
    set_wheels( v, -v, -v,  v)

def move_neg_y():
    set_wheels(-v,  v,  v, -v)

# helper functions to get the current (x, y) position of the robot and waypoints. These functions use sim.getObjectPosition to retrieve the position of the specified object in the world frame and return just the x and y coordinates.
def get_robot_xy():
    p = sim.getObjectPosition(robot, -1)
    return p[0], p[1]

def get_waypoint_xy(wp_handle):
    p = sim.getObjectPosition(wp_handle, -1)
    return p[0], p[1]

# axis aligned movement functions that run a control loop until the robot reaches the target coordinate within a specified tolerance. The move_along_x_to function moves the robot along the x-axis towards the target x coordinate, while the move_along_y_to function does the same for the y-axis. Both functions check for a stop request to allow for safe interruption, and they use the get_robot_xy function to continuously monitor the robot's position and adjust wheel velocities accordingly until the target is reached within tolerance.
def move_along_x_to(target_x):
    global stop_requested

    while True:
        if stop_requested:
            stop()
            return False

        x, _ = get_robot_xy()
        err_x = target_x - x

        if abs(err_x) <= pos_tolerance:
            stop()
            break

        if err_x > 0:
            move_pos_x()
        else:
            move_neg_x()

        time.sleep(control_dt)

    stop()
    time.sleep(settle_time)
    return True

def move_along_y_to(target_y):
    global stop_requested

    while True:
        if stop_requested:
            stop()
            return False

        _, y = get_robot_xy()
        err_y = target_y - y

        if abs(err_y) <= pos_tolerance:
            stop()
            break

        if err_y > 0:
            move_pos_y()
        else:
            move_neg_y()

        time.sleep(control_dt)

    stop()
    time.sleep(settle_time)
    return True

def move_to_waypoint_axis_aligned(wp_index, axis_order='xy'):
    wp_handle = waypoints[wp_index - 1]
    wx, wy = get_waypoint_xy(wp_handle)

    if axis_order == 'xy':
        ok = move_along_x_to(wx)
        if not ok:
            return False
        ok = move_along_y_to(wy)
        if not ok:
            return False
    elif axis_order == 'yx':
        ok = move_along_y_to(wy)
        if not ok:
            return False
        ok = move_along_x_to(wx)
        if not ok:
            return False
    else:
        raise ValueError("axis_order must be 'xy' or 'yx'")

    return True

def get_path(start_wp, target_wp):
    if start_wp < target_wp:
        return list(range(start_wp, target_wp + 1))
    elif start_wp > target_wp:
        return list(range(start_wp, target_wp - 1, -1))
    else:
        return [start_wp]

# helper functions to change plant colors. The set_plant_leaves_color function takes a plant name and an RGB color, retrieves the corresponding shape handle from the plant_shape_map, and uses sim.setShapeColor to change the ambient diffuse color of the shape to the specified RGB value. The set_plant_yellow and set_plant_green functions are convenience wrappers that set the plant color to yellow or green respectively and update the plant_is_yellow state. The toggle_plant_color function toggles the plant's color between yellow and green based on its current state.
def set_plant_leaves_color(plant_name, rgb):
    try:
        shape_handle = sim.getObject(plant_shape_map[plant_name])
        sim.setShapeColor(
            shape_handle,
            LEAF_COLOR_NAME,
            sim.colorcomponent_ambient_diffuse,
            rgb
        )
        return True
    except Exception as e:
        print(f"Could not change {plant_name}: {e}")
        return False

def set_plant_yellow(plant_name):
    ok = set_plant_leaves_color(plant_name, YELLOW)
    if ok:
        plant_is_yellow[plant_name] = True
    return ok

def set_plant_green(plant_name):
    ok = set_plant_leaves_color(plant_name, GREEN)
    if ok:
        plant_is_yellow[plant_name] = False
    return ok

def toggle_plant_color(plant_name):
    if plant_is_yellow[plant_name]:
        return set_plant_green(plant_name)
    else:
        return set_plant_yellow(plant_name)

def choose_best_waypoint(current_wp, candidate_wps):
    return min(candidate_wps, key=lambda wp: abs(wp - current_wp))

# aruco search and lock function. This function implements a practical lock logic with three states: SEARCH, TRACK, and LOCKED. In the SEARCH state, the robot sweeps its arm until it detects the ArUco marker. Once the marker is detected, it transitions to the TRACK state, where it tries to keep the marker in view by adjusting the arm joints based on the marker's position in the camera frame. If the marker remains visible for a certain number of consecutive frames (VISIBLE_HOLD_FRAMES), it transitions to the LOCKED state, indicating that the marker is successfully locked. The function also handles timeouts and allows for manual interruption by checking for a stop request.
def run_aruco_search_and_lock(timeout=ARUCO_LOCK_TIMEOUT):
    """
    Practical lock logic:
    - SEARCH: sweep until marker appears
    - TRACK: try to keep it in view
    - LOCK: if visible for VISIBLE_HOLD_FRAMES consecutive frames

    This is intentionally simpler and more robust than forcing the marker
    to hit one exact image location.
    """
    global stop_requested

    state = 'SEARCH'
    visible_hold = 0
    lost_count = 0

    ex_f = 0.0
    ey_f = 0.0

    set_joint_targets(home_q)
    time.sleep(1.0)

    set_joint_targets(search_q)
    time.sleep(1.0)

    start_time = time.time()

    while not stop_requested:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print("Aruco search timed out.")
            cv2.destroyWindow('UR5 ArUco Inspect')
            return False

        frame = get_sim_frame(vision_sensor)
        h, w = frame.shape[:2]

        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)
        marker_found = ids is not None and len(corners) > 0

        # Draw helper lines
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

        if marker_found:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if state == 'SEARCH':
            sweep_j1 = home_q[0] + SEARCH_SWEEP_AMPLITUDE * math.sin(SEARCH_SWEEP_SPEED * elapsed)

            sim.setJointTargetPosition(J1, clamp(sweep_j1, J1_MIN, J1_MAX))
            sim.setJointTargetPosition(J2, clamp(search_q[1], J2_MIN, J2_MAX))
            sim.setJointTargetPosition(J3, clamp(search_q[2], J3_MIN, J3_MAX))
            sim.setJointTargetPosition(J4, clamp(search_q[3], J4_MIN, J4_MAX))
            sim.setJointTargetPosition(J5, clamp(search_q[4], J5_MIN, J5_MAX))
            sim.setJointTargetPosition(J6, clamp(search_q[5], J6_MIN, J6_MAX))

            visible_hold = 0
            lost_count = 0

            if marker_found:
                state = 'TRACK'

            overlay = "STATE: SEARCH"

        elif state == 'TRACK':
            overlay = "STATE: TRACK"

            if marker_found:
                lost_count = 0

                c = corners[0].reshape(4, 2).astype(np.float32)
                mx = float(np.mean(c[:, 0]))
                my = float(np.mean(c[:, 1]))

                ex = (mx - (w / 2)) / (w / 2)
                ey = (my - (h / 2)) / (h / 2)

                ex_f = (1 - alpha) * ex_f + alpha * ex
                ey_f = (1 - alpha) * ey_f + alpha * ey

                q1 = sim.getJointPosition(J1)
                q2 = sim.getJointPosition(J2)

                # Keep marker near center; no strict lock target
                q1_cmd = q1 + arm_control_step * (SIGN_J1 * Kp_pan * ex_f)
                q2_cmd = q2 + arm_control_step * (SIGN_J2 * Kp_tilt * ey_f)

                q1_cmd = clamp(q1_cmd, J1_MIN, J1_MAX)
                q2_cmd = clamp(q2_cmd, J2_MIN, J2_MAX)

                sim.setJointTargetPosition(J1, q1_cmd)
                sim.setJointTargetPosition(J2, q2_cmd)
                sim.setJointTargetPosition(J3, clamp(search_q[2], J3_MIN, J3_MAX))
                sim.setJointTargetPosition(J4, clamp(search_q[3], J4_MIN, J4_MAX))
                sim.setJointTargetPosition(J5, clamp(search_q[4], J5_MIN, J5_MAX))
                sim.setJointTargetPosition(J6, clamp(search_q[5], J6_MIN, J6_MAX))

                visible_hold += 1
                overlay = f"STATE: TRACK  hold={visible_hold}/{VISIBLE_HOLD_FRAMES}  ex={ex_f:+.2f} ey={ey_f:+.2f}"

                # marker visible stably -> lock
                if visible_hold >= VISIBLE_HOLD_FRAMES:
                    state = 'LOCKED'

            else:
                visible_hold = 0
                lost_count += 1
                overlay = f"STATE: TRACK  lost={lost_count}"

                if lost_count > MAX_LOST_FRAMES:
                    state = 'SEARCH'
                    lost_count = 0

        else:  # LOCKED
            overlay = "STATE: LOCKED"
            cv2.putText(frame, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow('UR5 ArUco Inspect', frame)
            cv2.waitKey(1)
            time.sleep(0.3)
            cv2.destroyWindow('UR5 ArUco Inspect')
            return True

        cv2.putText(frame, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if ids is not None:
            ids_text = "IDs: " + ",".join(str(int(x[0])) for x in ids)
            cv2.putText(frame, ids_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        cv2.imshow('UR5 ArUco Inspect', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow('UR5 ArUco Inspect')
            return False

        time.sleep(1 / 30)

    cv2.destroyWindow('UR5 ArUco Inspect')
    return False

def inspect_and_restore_plant(plant_name):
    if not plant_is_yellow.get(plant_name, False):
        return

    print(f"At waypoint wp{current_wp}: inspecting {plant_name}...")

    found_and_locked = run_aruco_search_and_lock(timeout=ARUCO_LOCK_TIMEOUT)

    if found_and_locked:
        print(f"{plant_name}: ArUco found and locked. Restoring to green.")
        set_plant_green(plant_name)
    else:
        print(f"{plant_name}: ArUco not locked. Leaving plant yellow.")

def check_and_restore_plants_at_waypoint(wp_idx):
    plants_here = waypoint_plant_map.get(wp_idx, [])
    if not plants_here:
        return

    for plant_name in plants_here:
        if plant_is_yellow.get(plant_name, False):
            inspect_and_restore_plant(plant_name)

def move_through_path(path, axis_order='xy'):
    global current_wp

    for wp_idx in path[1:]:
        ok = move_to_waypoint_axis_aligned(wp_idx, axis_order=axis_order)
        if not ok:
            return False

        current_wp = wp_idx
        check_and_restore_plants_at_waypoint(current_wp)

    return True

# input worker thread that continuously prompts the user for commands and puts them into the command_queue. It handles commands for moving to waypoints, changing plant colors, and quitting the program. The worker runs in a loop until a quit command is received or a stop is requested, allowing the main thread to process commands concurrently while the robot is moving.
def input_worker():
    global stop_requested

    while not stop_requested:
        try:
            cmd = input("\nEnter command: ").strip().lower()
        except EOFError:
            cmd = 'q'
        except KeyboardInterrupt:
            cmd = 'q'

        if cmd == 'q':
            command_queue.put(('quit', None))
            break

        if cmd in plant_shape_map:
            if not plant_is_yellow[cmd]:
                set_plant_yellow(cmd)

            if cmd not in pending_plant_commands:
                pending_plant_commands.add(cmd)
                command_queue.put(('plant', cmd))
            continue

        if cmd.startswith("toggle "):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                print("Invalid plant name.")
                continue

            plant_name = parts[1]
            if plant_name in plant_shape_map:
                toggle_plant_color(plant_name)
            else:
                print("Invalid plant name.")
            continue

        if cmd.startswith("wp"):
            command_queue.put(('wp', cmd))
            continue

        print("Invalid command.")

# main control loop
sim.startSimulation()
time.sleep(0.5)

try:
    while True:
        try:
            current_wp = int(input("Enter current waypoint (1-10): ").strip())
            if 1 <= current_wp <= 10:
                break
            print("Waypoint must be between 1 and 10.")
        except ValueError:
            print("Please enter an integer.")

    print("\nCommands:")
    print("  p0 ... p13           -> make plant yellow immediately and queue robot movement")
    print("  wp1 ... wp10         -> queue direct waypoint movement")
    print("  toggle pX            -> toggle plant color only")
    print("  q                    -> quit")
    print("\nYou can keep entering commands while the robot is moving.")

    check_and_restore_plants_at_waypoint(current_wp)

    input_thread = threading.Thread(target=input_worker, daemon=True)
    input_thread.start()

    try:
        while not stop_requested:
            try:
                cmd_type, payload = command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if cmd_type == 'quit':
                stop_requested = True
                break

            if cmd_type == 'wp':
                cmd = payload
                try:
                    target_wp = int(cmd[2:])
                    if not (1 <= target_wp <= 10):
                        print("Invalid waypoint.")
                        continue

                    if target_wp == current_wp:
                        check_and_restore_plants_at_waypoint(current_wp)
                        continue

                    path = get_path(current_wp, target_wp)
                    move_through_path(path, axis_order='xy')

                except ValueError:
                    print("Invalid waypoint command.")
                continue

            if cmd_type == 'plant':
                plant_name = payload
                pending_plant_commands.discard(plant_name)

                if not plant_is_yellow.get(plant_name, False):
                    continue

                candidate_wps = plant_waypoint_map[plant_name]
                target_wp = choose_best_waypoint(current_wp, candidate_wps)

                if target_wp != current_wp:
                    path = get_path(current_wp, target_wp)
                    move_through_path(path, axis_order='xy')
                else:
                    check_and_restore_plants_at_waypoint(current_wp)

                continue

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping robot safely...")
        stop_requested = True

finally:
    stop_requested = True
    stop()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    sim.stopSimulation()
    print("Simulation stopped.")