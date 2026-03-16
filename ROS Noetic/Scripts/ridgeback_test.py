#!/usr/bin/env python3
import rospy
import actionlib
import json
import random
import time
import math
import sys
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, Quaternion
from nav_msgs.msg import Path
from std_srvs.srv import Empty
from tf.transformations import quaternion_from_euler

class RidgebackTester:
    def __init__(self, greenhouse_name):
        rospy.init_node('ridgeback_tester')
        self.filename = f"/home/lucas/catkin_ws/src/greenhouse_project/config/{greenhouse_name}.json"
        
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path = None
        
        rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)

        rospy.wait_for_service('/move_base/clear_costmaps')
        self.clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        
        self.plants = self.load_data()
        self.current_pose = None
        self.test_results = []

    def path_callback(self, data):
        self.global_path = data

    def pose_callback(self, data):
        self.current_pose = data.pose.pose

    def load_data(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            rospy.logerr(f"File {self.filename} not found!")
            sys.exit(1)

    def calculate_error(self, target_x, target_y):
        if self.current_pose:
            dx = target_x - self.current_pose.position.x
            dy = target_y - self.current_pose.position.y
            return math.sqrt(dx**2 + dy**2)
        return 999.9

    def get_path_deviation(self):
        """Measures distance to the blue global plan line."""
        if not self.global_path or not self.current_pose or len(self.global_path.poses) == 0:
            return 0.0
        min_dist = 999.9
        for p in self.global_path.poses:
            dx = p.pose.position.x - self.current_pose.position.x
            dy = p.pose.position.y - self.current_pose.position.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def run_test(self, num_samples=10):
        plant_names = list(self.plants.keys())
        random.shuffle(plant_names)
        test_queue = plant_names[:num_samples]

        # --- LOGIC THRESHOLDS ---
        MAX_DEVIATION = 0.40      # m 
        DRIFT_TIMEOUT = 60.0      # s 
        PRECISION_GOAL = 0.20     # m 
        MAX_TOTAL_TIME = 300.0    # s 

        idx = 0
        while idx < len(test_queue):
            name = test_queue[idx]
            data = self.plants[name]
            
            # Extract data from JSON plant nodes
            target_x = data.get('x', 0.0)
            target_y = data.get('y', 0.0)
            target_yaw = data.get('yaw', 0.0)

            rospy.loginfo(f"Target {idx+1}/{len(test_queue)}: {name} (Yaw: {target_yaw})")
            
            try:
                self.clear_costmaps()
            except: pass

            # Convert Euler Yaw to Quaternion
            q_list = quaternion_from_euler(0, 0, target_yaw)
            q = Quaternion(*q_list)

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = target_x
            goal.target_pose.pose.position.y = target_y
            goal.target_pose.pose.orientation = q 

            self.client.send_goal(goal)
            start_time = time.time()
            drift_start_time = None
            
            while not rospy.is_shutdown():
                state = self.client.get_state()
                error = self.calculate_error(target_x, target_y)
                deviation = self.get_path_deviation()
                elapsed_total = time.time() - start_time

                # SUCCESS CHECK
                if state == 3 or error < PRECISION_GOAL:
                    rospy.loginfo(f"SUCCESS: Reached {name} in {elapsed_total:.1f}s")
                    self.client.cancel_goal()
                    self.test_results.append({
                        "id": name, 
                        "status": "SUCCESS", 
                        "error": error, 
                        "time": elapsed_total
                    })
                    break

                # ABORTED CHECK (Status 4)
                if state == 4:
                    rospy.logwarn(f"ABORTED: ROS gave up on {name}. Re-queueing...")
                    test_queue.append(name)
                    break

                # PATH DEVIATION MONITOR
                if deviation > MAX_DEVIATION:
                    if drift_start_time is None:
                        drift_start_time = time.time()
                    
                    if (time.time() - drift_start_time) > DRIFT_TIMEOUT:
                        rospy.logwarn(f"DRIFT TIMEOUT: Off-path for {DRIFT_TIMEOUT}s. Re-queueing {name}...")
                        self.client.cancel_goal()
                        test_queue.append(name)
                        break
                else:
                    drift_start_time = None 

                # TIMEOUT
                if elapsed_total > MAX_TOTAL_TIME:
                    rospy.logerr(f"CRITICAL TIMEOUT: Goal {name} took too long. Moving on.")
                    self.client.cancel_goal()
                    self.test_results.append({
                        "id": name, 
                        "status": "FAIL_TIMEOUT", 
                        "error": error, 
                        "time": elapsed_total
                    })
                    break

                rospy.sleep(0.5)
            
            idx += 1

        # Save results
        output_path = "/home/lucas/catkin_ws/src/greenhouse_project/results_test.json"
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=4)
        rospy.loginfo("Test complete. Data saved.")

if __name__ == '__main__':
    gh_name = sys.argv[1] if len(sys.argv) > 1 else "greenhouse2"
    tester = RidgebackTester(gh_name)
    tester.run_test(num_samples=10)
