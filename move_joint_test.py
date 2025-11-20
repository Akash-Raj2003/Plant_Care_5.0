#!/usr/bin/env python3
import sys
import rospy
import moveit_commander

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("ur5_move_test", anonymous=True)

    group = moveit_commander.MoveGroupCommander("manipulator")

    rospy.loginfo("Moving joint 1 +0.2 rad...")

    joints = group.get_current_joint_values()
    joints[0] += 0.2
    joints[1] += 0.4
    joints[2] += 0.4

    group.go(joints, wait=True)
    group.stop()

    rospy.loginfo("Done.")
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()

