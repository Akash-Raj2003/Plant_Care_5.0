#!/usr/bin/env python3
import rospy
import json
import sys
import os
from geometry_msgs.msg import PoseWithCovarianceStamped

class AddPlant:

    def __init__(self, greenhouse_name):
        rospy.init_node('add_plant')
        
        self.config_dir = "/home/sope/catkin_ws/src/my_simulations/config"
        self.filename = os.path.join(self.config_dir, f"{greenhouse_name}.json")
        
        # Load existing data if file exists, else start fresh
        self.plant_db = self.load()
        
        self.sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.current_pose = None

    def load(self):
        """Checks if the greenhouse file exists and loads it."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    rospy.loginfo(f"Appending to existing file: {self.filename}")
                    return data
            except json.JSONDecodeError:
                rospy.logwarn("File was empty or corrupted. Starting fresh.")
        return {}

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose.position

    def record(self):
        print(f"--- Appending to: {self.filename} ---")
        while not rospy.is_shutdown():
            name = input("\nEnter Plant Name to add (or 'save' to exit): ").strip()
            if name.lower() == 'save':
                self.save()
                break
            
            if self.current_pose:
                # Add or update the plant entry in the dictionary
                self.plant_db[name] = {
                    "plant_id": len(self.plant_db) + 1,
                    "x": round(self.current_pose.x, 3),
                    "y": round(self.current_pose.y, 3),
                    "z": round(self.current_pose.z, 3),
                    "moisture_level": 0.0,
                    "moisture_alert": False
                }
                print(f"Added {name}. Total plants in library: {len(self.plant_db)}")
            else:
                print("Waiting for AMCL localization...")

    def save(self):
        # Create directory if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        with open(self.filename, 'w') as f:
            json.dump(self.plant_db, f, indent=4)
            
        print(f"Saved {len(self.plant_db)} plants to {self.filename}")

if __name__ == '__main__':
    # Usage: rosrun my_simulations add_plant4.py greenhouse2
    gh_name = sys.argv[1] if len(sys.argv) > 1 else "greenhouse_1"
    adder = AddPlant(gh_name)
    adder.record()
