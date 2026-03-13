from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

# -----------------------------
# Connect
# -----------------------------
client = RemoteAPIClient()
sim = client.getObject('sim')

# -----------------------------
# Wheel joints
# -----------------------------
joint_paths = [
    '/OmniPlatform/link[0]/regularRotation',
    '/OmniPlatform/link[1]/regularRotation',
    '/OmniPlatform/link[2]/regularRotation',
    '/OmniPlatform/link[3]/regularRotation',
]
joints = [sim.getObject(p) for p in joint_paths]

# -----------------------------
# Robot + waypoint handles
# -----------------------------
robot = sim.getObject('/OmniPlatform')

waypoint_paths = [
    '/wp1',
    '/wp2',
    '/wp3',
    '/wp4',
    '/wp5',
    '/wp6',
    '/wp7',
    '/wp8',
    '/wp9',
    '/wp10',
]
waypoints = [sim.getObject(p) for p in waypoint_paths]

# -----------------------------
# Tunable parameters
# -----------------------------
v = 100 * 2.398795 * math.pi / 180.0
pos_tolerance = 0.10
control_dt = 0.05
settle_time = 0.2

# -----------------------------
# Wheel motion patterns
# Verified:
# +X = [-v, -v, +v, +v]
# -X = [+v, +v, -v, -v]
# +Y = [+v, -v, -v, +v]
# -Y = [-v, +v, +v, -v]
# -----------------------------
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

def get_robot_xy():
    p = sim.getObjectPosition(robot, -1)
    return p[0], p[1]

def get_waypoint_xy(wp_handle):
    p = sim.getObjectPosition(wp_handle, -1)
    return p[0], p[1]

def move_along_x_to(target_x):
    while True:
        x, y = get_robot_xy()
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

def move_along_y_to(target_y):
    while True:
        x, y = get_robot_xy()
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

def move_to_waypoint_axis_aligned(wp_index, axis_order='xy'):
    wp_handle = waypoints[wp_index - 1]
    wx, wy = get_waypoint_xy(wp_handle)

    print(f"Moving to wp{wp_index} at x={wx:.3f}, y={wy:.3f}")

    if axis_order == 'xy':
        move_along_x_to(wx)
        move_along_y_to(wy)
    elif axis_order == 'yx':
        move_along_y_to(wy)
        move_along_x_to(wx)
    else:
        raise ValueError("axis_order must be 'xy' or 'yx'")

    rx, ry = get_robot_xy()
    print(f"Reached near wp{wp_index}. Robot at x={rx:.3f}, y={ry:.3f}")

def get_path(current_wp, target_wp):
    if current_wp < target_wp:
        return list(range(current_wp, target_wp + 1))
    elif current_wp > target_wp:
        return list(range(current_wp, target_wp - 1, -1))
    else:
        return [current_wp]

def move_through_path(path, axis_order='xy'):
    # Skip the first waypoint because that is the current location
    for wp_idx in path[1:]:
        print(f"\n--- Going to wp{wp_idx} ---")
        move_to_waypoint_axis_aligned(wp_idx, axis_order=axis_order)

def is_valid_wp(n):
    return 1 <= n <= len(waypoints)

# -----------------------------
# Main
# -----------------------------
sim.startSimulation()
time.sleep(0.5)

try:
    # Ask initial current waypoint once
    while True:
        try:
            current_wp = int(input(f"Enter current waypoint (1-{len(waypoints)}): "))
            if is_valid_wp(current_wp):
                break
            print("Invalid waypoint number.")
        except ValueError:
            print("Please enter an integer.")

    while True:
        user_in = input(
            f"\nEnter target waypoint (1-{len(waypoints)}) or 'q' to quit: "
        ).strip().lower()

        if user_in == 'q':
            break

        try:
            target_wp = int(user_in)
        except ValueError:
            print("Please enter an integer or q.")
            continue

        if not is_valid_wp(target_wp):
            print("Invalid waypoint number.")
            continue

        if target_wp == current_wp:
            print(f"Robot is already at wp{current_wp}.")
            continue

        path = get_path(current_wp, target_wp)
        print("Planned path:", " -> ".join([f"wp{i}" for i in path]))

        move_through_path(path, axis_order='xy')

        current_wp = target_wp
        print(f"\nCurrent waypoint updated to wp{current_wp}")

    stop()
    time.sleep(0.5)

finally:
    stop()
    sim.stopSimulation()