#!/usr/bin/env python3

from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
from plant_msgs.msg import PlantMoisture
from std_msgs.msg import Int32


plant_dict = {
    1 : (1.5, 2.0),
    2 : (-3, -10),
    3 : (0, -6),
    4 : (3, -10)
}

robot_state = 0

#States for ridgeback
available = 0
moving_to_plant = 1
waiting_for_dispense = 2
dispensing = 3
complete = 4


waypoint_publisher = None
state_publisher = None
plant_queue = deque()
queued_plants = set()
auto_dispatch_queue = True


def create_waypoint(plant):
    # Create a PoseStamped waypoint for the given plant name.

    if plant in plant_dict:
        x, y = plant_dict[plant]

        waypoint = PoseStamped()
        waypoint.header.frame_id = "map"
        waypoint.header.stamp = rospy.Time.now()

        waypoint.pose.position.x = x
        waypoint.pose.position.y = y + 2   # offset so robot stops before plant
        waypoint.pose.position.z = 0.0

        waypoint.pose.orientation.x = 0.0
        waypoint.pose.orientation.y = 0.0
        waypoint.pose.orientation.z = 0.0
        waypoint.pose.orientation.w = 1.0

        return waypoint
    else:
        rospy.logwarn("Plant '%s' not found in evironment.", plant)
        return None


def queue_plant(plant_id):
    if plant_id in queued_plants:
        rospy.loginfo("Plant %d is already queued. Current queue: %s", plant_id, list(plant_queue))
        return False

    plant_queue.append(plant_id)
    queued_plants.add(plant_id)
    rospy.loginfo("Queued plant %d. Current queue: %s", plant_id, list(plant_queue))
    return True


def publish_next_waypoint():
    if not plant_queue:
        rospy.loginfo("Plant queue is empty. No waypoint published.")
        return None

    plant_id = plant_queue[0]
    waypoint = create_waypoint(plant_id)

    if waypoint is None:
        plant_queue.popleft()
        queued_plants.discard(plant_id)
        rospy.logwarn("Removed plant %d from queue because no waypoint was found.", plant_id)
        return None

    waypoint_publisher.publish(waypoint)
    rospy.loginfo("Published waypoint for queued plant %d", plant_id)
    return plant_id


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

        plant_was_added = queue_plant(plant_id)
        queue_was_empty_before_add = len(plant_queue) == 1

        # If auto dispatch is enabled, send the first queued plant immediately.
        # Later plants stay queued until another part of the state logic asks for
        # the next waypoint to be published.
        if auto_dispatch_queue and plant_was_added and queue_was_empty_before_add:
            publish_next_waypoint()


if __name__ == '__main__':
    rospy.init_node('waypoint_sender_node')
    rospy.loginfo("Waypoint sender node started.")

    auto_dispatch_queue = rospy.get_param("~auto_dispatch_queue", True)
    waypoint_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    state_publisher = rospy.Publisher('/ridgeback/state', Int32, queue_size=10)

    plantinfo_subscriber = rospy.Subscriber(
        "/plant/moisture_alert",
        PlantMoisture,
        plant_callback
    )

    rospy.loginfo("Plant queue ready. auto_dispatch_queue=%s", auto_dispatch_queue)
    rospy.spin()
