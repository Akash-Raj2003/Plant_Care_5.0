#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def main():
    # ----- INIT ROS + MOVEIT -----
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("ur5_move_to_pose", anonymous=True)

    # UR5 arm planning group
    group = moveit_commander.MoveGroupCommander("manipulator")

    rospy.loginfo("Current pose:")
    rospy.loginfo(group.get_current_pose())

    # ----- TARGET POSE -----
    pose_target = Pose()
    pose_target.position.x = 0.4
    pose_target.position.y = 0.0
    pose_target.position.z = 0.3

    # simple orientation facing forward
    pose_target.orientation.w = 1.0

    rospy.loginfo("Setting pose target...")
    group.set_pose_target(pose_target)

    # ----- PLAN + EXECUTE -----
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    rospy.loginfo("Movement complete.")
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()

