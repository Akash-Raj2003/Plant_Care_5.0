#!/usr/bin/env python3
import rospy
import actionlib
import json
import random
import time
import math
import sys
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty # NEW: Required for clearing costmaps

class RidgebackTester:
    def __init__(self, greenhouse_name):
        rospy.init_node('ridgeback_tester')
        
        self.filename = f"/home/lucas/catkin_ws/src/greenhouse_project/config/{greenhouse_name}.json"
        
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()

        # NEW: Initialize the service proxy to clear costmaps
        rospy.loginfo("Waiting for costmap clearing service...")
        rospy.wait_for_service('/move_base/clear_costmaps')
        self.clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        
        self.plants = self.load_data()
        self.current_pose = None
        self.test_results = []
        
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)

    def load_data(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            rospy.logerr(f"File {self.filename} not found!")
            sys.exit(1)

    def pose_callback(self, data):
        self.current_pose = data.pose.pose

    def calculate_error(self, target_x, target_y):
        if self.current_pose:
            dx = target_x - self.current_pose.position.x
            dy = target_y - self.current_pose.position.y
            return math.sqrt(dx**2 + dy**2)
        return None

    def run_test(self, num_samples=10):
        plant_names = list(self.plants.keys())
        random.shuffle(plant_names)
        test_queue = plant_names[:num_samples]

        for name in test_queue:
            # NEW: Clear costmaps before every new goal to wipe "ghost" obstacles
            rospy.loginfo(f"Clearing costmaps and heading to: {name}")
            try:
                self.clear_costmaps()
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to clear costmaps: {e}")

            data = self.plants[name]
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = data['x']
            goal.target_pose.pose.position.y = data['y']
            goal.target_pose.pose.orientation.w = 1.0

            start_time = time.time()
            self.client.send_goal(goal)
            
            # Wait for 60 seconds. If it fails, we move to the next sample.
            self.client.wait_for_result(rospy.Duration(60.0))
            
            duration = time.time() - start_time
            error = self.calculate_error(data['x'], data['y'])
            state = self.client.get_state()
            
            result = {
                "plant_id": name,
                "travel_time_s": round(duration, 2),
                "distance_error_m": round(error, 4) if error is not None else "N/A",
                "reached": True if state == 3 else False,
                "action_status": state
            }
            
            self.test_results.append(result)
            rospy.loginfo(f"Result for {name}: {result}")

        # Save results
        output_path = f"/home/lucas/catkin_ws/src/greenhouse_project/results_test.json"
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=4)
        rospy.loginfo(f"Test complete. Results saved to {output_path}")

if __name__ == '__main__':
    gh_name = sys.argv[1] if len(sys.argv) > 1 else "greenhouse2"
    tester = RidgebackTester(gh_name)
    tester.run_test(num_samples=10)
