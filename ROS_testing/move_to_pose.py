#!/usr/bin/env python3
import sys #import dependencies
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def main():
    # ----- INIT ROS + MOVEIT -----
    moveit_commander.roscpp_initialize(sys.argv) #start moveit commander
    rospy.init_node("ur5_move_to_pose", anonymous=True) #creates node ur5_move_to_pose

    # UR5 arm planning group
    group = moveit_commander.MoveGroupCommander("manipulator")

    rospy.loginfo("Current pose:") #print current end effector pose
    rospy.loginfo(group.get_current_pose())

    # ----- TARGET POSE -----
    pose_target = Pose() #creates x y z pose (relative to base_link)
    pose_target.position.x = 0.4
    pose_target.position.y = 0.0
    pose_target.position.z = 0.3

    # simple orientation facing forward
    pose_target.orientation.w = 1.0 #means no rotation if set to 1

    rospy.loginfo("Setting pose target...")
    group.set_pose_target(pose_target)

    # ----- PLAN + EXECUTE -----
    plan = group.go(wait=True) #executes movement
    group.stop()
    group.clear_pose_targets()

    rospy.loginfo("Movement complete.") #print completion
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()

