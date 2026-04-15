#!/usr/bin/env python3

from collections import deque
import json
import math
import os

import actionlib
import rospy
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from my_simulations.msg import dispensingAction, dispensingGoal
from plant_msgs.msg import PlantMoisture
from std_msgs.msg import Int32, Int32MultiArray, String
import random

plant_dict = {}

robot_state = 0

#States for ridgeback
available = 0
moving_to_plant = 1
waiting_for_dispense = 2
dispensing = 3
complete = 4
interrupted = 5


# waypoint_publisher = None
move_base_client = None
state_publisher = None
queue_publisher = None
plant_queue = deque()
queued_plants = set()
auto_dispatch = True
dispense_client = None
gui_command_subscriber = None
emergency_stop_active = False
current_plant_id = None

GUI_COMMAND_TOPIC = "/plantcare/gui_command"
GUI_EMERGENCY_STOP = "emergency_stop"
GUI_RESUME_DISPATCH = "resume_dispatch"


def load_plant_dict(config_path):
    with open(config_path, "r") as config_file:
        plant_data = json.load(config_file)

    loaded_plants = {}
    for plant_name, plant_info in plant_data.items():
        plant_id = plant_info.get("plant_id")
        if plant_id is None:
            rospy.logwarn("Skipping %s because plant_id is missing.", plant_name)
            continue

        loaded_plants[int(plant_id)] = {
            "x": float(plant_info["x"]),
            "y": float(plant_info["y"]),
            "z": float(plant_info.get("z", 0.0)),
            "yaw": float(plant_info.get("yaw", 0.0)),
        }

    return loaded_plants


def set_robot_state(state):
    global robot_state
    robot_state = state
    state_publisher.publish(state)


def publish_plant_queue():
    queue_msg = Int32MultiArray()
    queue_msg.data = list(plant_queue)
    queue_publisher.publish(queue_msg)


def restore_current_plant_to_queue():
    global current_plant_id

    if current_plant_id is None or current_plant_id in queued_plants:
        return

    plant_queue.appendleft(current_plant_id)
    queued_plants.add(current_plant_id)
    publish_plant_queue()
    rospy.loginfo("Restored plant %d to front of queue after interruption.", current_plant_id)


def handle_emergency_stop():
    global auto_dispatch, emergency_stop_active

    rospy.logwarn("Emergency stop requested from GUI.")
    emergency_stop_active = True
    auto_dispatch = False

    if move_base_client is not None:
        move_base_client.cancel_all_goals()

    if dispense_client is not None:
        dispense_client.cancel_all_goals()

    restore_current_plant_to_queue()
    set_robot_state(interrupted)


def handle_resume_dispatch():
    global auto_dispatch, emergency_stop_active

    rospy.loginfo("Resume dispatch requested from GUI.")
    emergency_stop_active = False
    auto_dispatch = True
    set_robot_state(available)


def gui_command_callback(msg):
    if msg.data == GUI_EMERGENCY_STOP:
        handle_emergency_stop()
    elif msg.data == GUI_RESUME_DISPATCH:
        handle_resume_dispatch()


def create_waypoint(plant):
    # Create a PoseStamped waypoint for the given plant name.

    if plant in plant_dict:
        plant_pose = plant_dict[plant]
        yaw = plant_pose["yaw"]
        target_y = 5.6 if plant_pose["y"] >= 3.0 else 0.7

        waypoint = PoseStamped()
        waypoint.header.frame_id = "map"
        waypoint.header.stamp = rospy.Time.now()

        waypoint.pose.position.x = plant_pose["x"]
        waypoint.pose.position.y = target_y
        waypoint.pose.position.z = plant_pose["z"]

        waypoint.pose.orientation.x = 0.0
        waypoint.pose.orientation.y = 0.0
        waypoint.pose.orientation.z = math.sin(yaw / 2.0)
        waypoint.pose.orientation.w = math.cos(yaw / 2.0)

        return waypoint
    else:
        rospy.logwarn("Plant '%s' not found in evironment.", plant)
        return None


def dispense_feedback_cb(feedback):
    rospy.loginfo(
        "Dispensing progress: %.1f%% (%s)",
        feedback.percent_complete,
        feedback.state
    )


def dispense_for_plant(plant_id):
    global emergency_stop_active, current_plant_id
    goal = dispensingGoal()
    goal.plant_id = plant_id
    goal.target_volume = random.uniform(30.0, 70.0)

    set_robot_state(dispensing)
    rospy.loginfo(
        "Starting dispense for plant %d with target volume %.1f ml",
        plant_id,
        goal.target_volume
    )
    dispense_client.send_goal(goal, feedback_cb=dispense_feedback_cb)
    dispense_client.wait_for_result()

    result = dispense_client.get_result()
    action_state = dispense_client.get_state()

    if action_state == actionlib.GoalStatus.PREEMPTED:
        rospy.logwarn("Dispensing cancelled for plant %d", plant_id)
        if emergency_stop_active:
            restore_current_plant_to_queue()
            set_robot_state(interrupted)
        else:
            set_robot_state(available)
        return

    if result and result.success:
        rospy.loginfo(
            "Dispensing complete for plant %d: %.2f ml",
            plant_id,
            result.actual_volume
        )
    else:
        rospy.logwarn("Dispensing failed for plant %d", plant_id)

    set_robot_state(complete)
    set_robot_state(available)
    emergency_stop_active = False
    current_plant_id = None


def queue_plant(plant_id):
    if plant_id in queued_plants:
        rospy.loginfo("Plant %d is already queued. Current queue: %s", plant_id, list(plant_queue))
        return False

    plant_queue.append(plant_id)
    queued_plants.add(plant_id)
    publish_plant_queue()
    rospy.loginfo("Queued plant %d. Current queue: %s", plant_id, list(plant_queue))
    return True


def remove_plant_from_queue(plant_id):
    if plant_queue and plant_queue[0] == plant_id:
        plant_queue.popleft()
    else:
        try:
            plant_queue.remove(plant_id)
        except ValueError:
            pass

    queued_plants.discard(plant_id)
    publish_plant_queue()
    rospy.loginfo("Removed plant %d from queue. Current queue: %s", plant_id, list(plant_queue))


def send_next_waypoint():
    global emergency_stop_active, current_plant_id
    if robot_state != available:
        rospy.loginfo("Robot is not available. Current state: %d", robot_state)
        return None

    if not plant_queue:
        rospy.loginfo("Plant queue is empty. No waypoint published.")
        set_robot_state(available)
        return None

    plant_id = plant_queue[0]
    current_plant_id = plant_id
    waypoint = create_waypoint(plant_id)

    if waypoint is None:
        remove_plant_from_queue(plant_id)
        rospy.logwarn("Removed plant %d from queue because no waypoint was found.", plant_id)
        return None

    # waypoint_publisher.publish(waypoint)
    goal = MoveBaseGoal() # create goal object
    goal.target_pose = waypoint #set the goal's target pose to the waypoint

    set_robot_state(moving_to_plant) 
    move_base_client.send_goal(goal) 
    rospy.loginfo("Sent move_base goal for queued plant %d", plant_id)
    rospy.loginfo("Current queue after dispatch: %s", list(plant_queue))
    
    move_base_client.wait_for_result()
    result = move_base_client.get_state()
    
    if result == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Goal reached for plant %d", plant_id)
        remove_plant_from_queue(plant_id)
        set_robot_state(waiting_for_dispense)
        rospy.loginfo("Ridgeback waiting_for_dispense for plant %d", plant_id)
        dispense_for_plant(plant_id)
    elif result == actionlib.GoalStatus.PREEMPTED:
        rospy.logwarn("Navigation cancelled for plant %d", plant_id)
        if emergency_stop_active:
            set_robot_state(interrupted)
        else:
            set_robot_state(available)
    else:
        rospy.logwarn("Navigation failed for plant %d", plant_id)
        remove_plant_from_queue(plant_id)
        set_robot_state(available)
        current_plant_id = None

    return plant_id


def dispatch_timer_callback(_event):
    if not auto_dispatch:
        return

    if emergency_stop_active:
        return

    if robot_state != available:
        return

    if not plant_queue:
        return

    send_next_waypoint()
    

def plant_callback(msg):
    # Callback triggered whenever plant moisture data is received.
    rospy.loginfo(
        "Received moisture alert for plant_%d: Moisture level = %.2f, Low moisture = %s",
        msg.plant_id,
        msg.moisture_level,
        msg.low_moisture
    )
    
    if msg.low_moisture:
        plant_id = msg.plant_id
        rospy.loginfo("Plant %d has low moisture. Adding request to queue.", plant_id)

        queue_plant(plant_id)


if __name__ == '__main__':
    rospy.init_node('waypoint_sender_node')
    rospy.loginfo("Waypoint sender node started.")

    auto_dispatch = rospy.get_param("~auto_dispatch", True)
    default_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'config', 'greenhouse2.json')
    )
    plant_config_path = rospy.get_param("~plant_config", default_config_path)
    plant_dict = load_plant_dict(plant_config_path)
    rospy.loginfo("Loaded %d plant waypoints from %s", len(plant_dict), plant_config_path)
    # waypoint_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    dispense_client = actionlib.SimpleActionClient('dispense_water', dispensingAction)
    move_base_client.wait_for_server()
    dispense_client.wait_for_server()
    state_publisher = rospy.Publisher('/ridgeback/state', Int32, queue_size=10)
    queue_publisher = rospy.Publisher('plant/queue', Int32MultiArray, queue_size=10)
    set_robot_state(available)
    publish_plant_queue()

    plantinfo_subscriber = rospy.Subscriber(
        "/plant/moisture_alert",
        PlantMoisture,
        plant_callback
    )
    gui_command_subscriber = rospy.Subscriber(
        GUI_COMMAND_TOPIC,
        String,
        gui_command_callback,
    )
    dispatch_timer = rospy.Timer(rospy.Duration(0.2), dispatch_timer_callback)

    rospy.loginfo("Plant queue ready. auto_dispatch=%s", auto_dispatch)
    rospy.spin()
