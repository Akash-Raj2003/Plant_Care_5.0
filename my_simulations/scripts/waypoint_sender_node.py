#!/usr/bin/env python3

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


def send_waypoint(plant):
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
        rospy.loginfo("Plant %d has low moisture. Sending waypoint to Ridgeback.", plant_id)

        waypoint = send_waypoint(plant_id)

        if waypoint is not None:
            waypoint_publisher.publish(waypoint)
            rospy.loginfo("Published waypoint for Plant %d", plant_id)


if __name__ == '__main__':
    rospy.init_node('waypoint_sender_node')
    rospy.loginfo("Waypoint sender node started.")

    waypoint_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    state_publisher = rospy.Publisher('/ridgeback/state', Int32, queue_size=10)

    plantinfo_subscriber = rospy.Subscriber(
        "/plant/moisture_alert",
        PlantMoisture,
        plant_callback
    )

    rospy.spin()