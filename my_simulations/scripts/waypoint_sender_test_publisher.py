#!/usr/bin/env python3

import json
import os
import random

import rospy

from plant_msgs.msg import PlantMoisture


def load_default_plant_ids():
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "greenhouse2.json")
    )

    with open(config_path, "r") as config_file:
        plant_data = json.load(config_file)

    plant_ids = sorted(int(plant_info["plant_id"]) for plant_info in plant_data.values())
    return config_path, plant_ids


def main():
    rospy.init_node("waypoint_sender_test_publisher")

    pub = rospy.Publisher("/plant/moisture_alert", PlantMoisture, queue_size=10)

    config_path, default_plant_ids = load_default_plant_ids()
    plant_ids = rospy.get_param("~plant_ids", default_plant_ids)
    moisture_level = rospy.get_param("~moisture_level", 10.0)
    publish_interval = rospy.get_param("~publish_interval", 30.0)
    publish_count = rospy.get_param("~publish_count", 10)

    if not plant_ids:
        rospy.logerr("No plant IDs configured. Set ~plant_ids to a non-empty list.")
        return

    rate = rospy.Rate(1.0 / publish_interval)
    rospy.loginfo(
        "Publishing %d plant moisture alerts to /plant/moisture_alert every %.1f seconds using %d plant IDs from %s",
        publish_count,
        publish_interval,
        len(plant_ids),
        config_path,
    )

    for _ in range(int(publish_count)):
        if rospy.is_shutdown():
            break

        msg = PlantMoisture()
        msg.plant_id = int(random.choice(plant_ids))
        msg.moisture_level = float(moisture_level)
        msg.low_moisture = True

        pub.publish(msg)
        rospy.loginfo(
            "Published test moisture alert for plant %d with moisture %.2f",
            msg.plant_id,
            msg.moisture_level,
        )

        rate.sleep()

    rospy.loginfo("Finished publishing %d test moisture alerts.", int(publish_count))


if __name__ == "__main__":
    main()
