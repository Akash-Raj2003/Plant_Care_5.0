#!/usr/bin/env python3
import rospy
import json
import sys
import os
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion # Added for yaw conversion

class AddPlant:

    def __init__(self, greenhouse_name):
        rospy.init_node('add_plant')
        
        self.config_dir = "/home/lucas/catkin_ws/src/greenhouse_project/config"
        self.filename = os.path.join(self.config_dir, f"{greenhouse_name}.json")
        
        # Load existing data if file exists, else start fresh
        self.plant_db = self.load()
        
        self.sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        self.current_pose = None # This will now store the full pose message

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
        # Store the entire pose so we can access position AND orientation
        self.current_pose = msg.pose.pose

    def record(self):
        print(f"--- Appending to: {self.filename} ---")
        while not rospy.is_shutdown():
            name = input("\nEnter Plant Name to add (or 'save' to exit): ").strip()
            if name.lower() == 'save':
                self.save()
                break
            
            if self.current_pose:
                # 1. Extract Quaternion from the pose
                orient = self.current_pose.orientation
                quaternion = (orient.x, orient.y, orient.z, orient.w)
                
                # 2. Convert Quaternion to Euler (Roll, Pitch, Yaw)
                # We only care about the 3rd value (Yaw)
                _, _, yaw = euler_from_quaternion(quaternion)

                # 3. Add or update the plant entry in the dictionary
                self.plant_db[name] = {
                    "plant_id": len(self.plant_db) + 1,
                    "x": round(self.current_pose.position.x, 3),
                    "y": round(self.current_pose.position.y, 3),
                    "z": round(self.current_pose.position.z, 3),
                    "yaw": round(yaw, 3), # Added Yaw here
                    "moisture_level": 0.0,
                    "moisture_alert": False
                }
                print(f"Added {name} at Yaw: {round(yaw, 3)}. Total: {len(self.plant_db)}")
            else:
                print("Waiting for AMCL localization...")

    def save(self):
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        with open(self.filename, 'w') as f:
            json.dump(self.plant_db, f, indent=4)
            
        print(f"Saved {len(self.plant_db)} plants to {self.filename}")

if __name__ == '__main__':
    gh_name = sys.argv[1] if len(sys.argv) > 1 else "greenhouse_1"
    adder = AddPlant(gh_name)
    adder.record()
