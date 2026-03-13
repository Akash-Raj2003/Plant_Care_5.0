from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import threading
import queue

#connect to CoppeliaSim's remote API server
client = RemoteAPIClient()
sim = client.getObject('sim')

# handles for the omni directional robot's joints and waypoints in the environment
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

# control parameters
v = 100 * 2.398795 * 3.141592653589793 / 180.0
pos_tolerance = 0.10
control_dt = 0.05
settle_time = 0.2

# plant mapping and state
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

# Reverse lookup: waypoint -> list of plants at that waypoint
waypoint_plant_map = {i: [] for i in range(1, 11)}
for plant_name, wp_list in plant_waypoint_map.items():
    for wp in wp_list:
        waypoint_plant_map[wp].append(plant_name)

# state for command processing
command_queue = queue.Queue()
stop_requested = False
current_wp = None
pending_plant_commands = set()

#operation helper functions
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

#helper functions to get positions
def get_robot_xy():
    p = sim.getObjectPosition(robot, -1)
    return p[0], p[1]

def get_waypoint_xy(wp_handle):
    p = sim.getObjectPosition(wp_handle, -1)
    return p[0], p[1]

# movement functions that run a control loop until the robot reaches the target coordinate within tolerance.
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

    # print(f"Moving to wp{wp_index}")

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

# plant color control functions
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
        # print(f"{plant_name} changed to yellow")
    return ok

def set_plant_green(plant_name):
    ok = set_plant_leaves_color(plant_name, GREEN)
    if ok:
        plant_is_yellow[plant_name] = False
        # print(f"{plant_name} restored to green")
    return ok

def toggle_plant_color(plant_name):
    if plant_is_yellow[plant_name]:
        return set_plant_green(plant_name)
    else:
        return set_plant_yellow(plant_name)

def choose_best_waypoint(current_wp, candidate_wps):
    return min(candidate_wps, key=lambda wp: abs(wp - current_wp))

def check_and_restore_plants_at_waypoint(wp_idx):
    """
    If any plant mapped to this waypoint is currently yellow,
    restore it to green when the robot reaches this waypoint.
    """
    plants_here = waypoint_plant_map.get(wp_idx, [])
    if not plants_here:
        return

    restored_any = False
    for plant_name in plants_here:
        if plant_is_yellow.get(plant_name, False):
            # print(f"wp{wp_idx}: found yellow plant {plant_name}, restoring to green")
            set_plant_green(plant_name)
            restored_any = True

    # if not restored_any:
    #     print(f"wp{wp_idx}: no yellow plants to restore")

def move_through_path(path, axis_order='xy'):
    global current_wp

    for wp_idx in path[1:]:
        # print(f"--- Going to wp{wp_idx} ---")
        ok = move_to_waypoint_axis_aligned(wp_idx, axis_order=axis_order)
        if not ok:
            return False

        current_wp = wp_idx

        # As soon as robot reaches a waypoint, restore any yellow plants there
        check_and_restore_plants_at_waypoint(current_wp)

    return True

# thread function to read user input without blocking the main control loop
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

        # direct plant request: mark yellow immediately, queue movement
        if cmd in plant_shape_map:
            if not plant_is_yellow[cmd]:
                set_plant_yellow(cmd)
            # else:
            #     print(f"{cmd} is already yellow")

            if cmd not in pending_plant_commands:
                pending_plant_commands.add(cmd)
                command_queue.put(('plant', cmd))
            # else:
            #     print(f"{cmd} is already queued")

            continue

        # toggle-only command
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

        # direct waypoint request
        if cmd.startswith("wp"):
            command_queue.put(('wp', cmd))
            continue

        print("Invalid command.")

#main control loop
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

    # Check the starting waypoint once
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
                        # print(f"Robot is already at wp{current_wp}.")
                        check_and_restore_plants_at_waypoint(current_wp)
                        continue

                    path = get_path(current_wp, target_wp)
                    # print("Planned path:", " -> ".join(f"wp{i}" for i in path))
                    move_through_path(path, axis_order='xy')

                except ValueError:
                    print("Invalid waypoint command.")
                continue

            if cmd_type == 'plant':
                plant_name = payload
                pending_plant_commands.discard(plant_name)

                # If already restored on the way to somewhere else, skip it
                if not plant_is_yellow.get(plant_name, False):
                    # print(f"{plant_name} is already green. Skipping queued visit.")
                    continue

                candidate_wps = plant_waypoint_map[plant_name]
                target_wp = choose_best_waypoint(current_wp, candidate_wps)

                # print(f"{plant_name} is mapped to waypoint(s) {candidate_wps}.")
                # print(f"Chosen target waypoint: wp{target_wp}")

                if target_wp != current_wp:
                    path = get_path(current_wp, target_wp)
                    # print("Planned path:", " -> ".join(f"wp{i}" for i in path))
                    move_through_path(path, axis_order='xy')
                else:
                    # print(f"Robot is already at wp{current_wp}.")
                    check_and_restore_plants_at_waypoint(current_wp)

                continue

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping robot safely...")
        stop_requested = True

finally:
    stop_requested = True
    stop()
    sim.stopSimulation()
    print("Simulation stopped.")