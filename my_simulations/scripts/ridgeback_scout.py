#!/usr/bin/env python3
import rospy
import actionlib
import json
import sys
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class GreenhouseScout:
    def __init__(self, greenhouse_name):
        rospy.init_node('greenhouse_scout')
        self.filename = f"/home/lucas/catkin_ws/src/greenhouse_project/config/{greenhouse_name}.json"
        
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        
        self.plants = self.load_data()

    def load_data(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                rospy.loginfo(f"Loaded greenhouse: {self.filename}")
                return data
        except FileNotFoundError:
            rospy.logerr(f"File {self.filename} not found!")
            sys.exit(1)

    def run(self):
        for name, data in self.plants.items():
            rospy.loginfo(f"Navigating to {name}...")
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.pose.position.x = data['x']
            goal.target_pose.pose.position.y = data['y']
            goal.target_pose.pose.orientation.w = 1.0
            
            self.client.send_goal_and_wait(goal)

if __name__ == '__main__':
    # Usage: rosrun greenhouse_project scout.py greenhouse_A
    name = sys.argv[1] if len(sys.argv) > 1 else "default_greenhouse"
    scout = GreenhouseScout(name)
    scout.run()
