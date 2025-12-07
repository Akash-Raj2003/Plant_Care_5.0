#!/usr/bin/env python3
import sys #import dependencies
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def create_pose(x, y, z):
    """Helper to quickly create a Pose with identity orientation."""
    p = Pose()
    p.position.x = x #creates position from parsed local variables
    p.position.y = y
    p.position.z = z
    p.orientation.w = 1.0  # simple orientation
    return p

def main():
    # ----- INIT ROS + MOVEIT -----
    moveit_commander.roscpp_initialize(sys.argv) #start moveit commander
    rospy.init_node("ur5_chained_moves", anonymous=True) #creates node ur5_move_to_pose

    # UR5 arm planning group
    group = moveit_commander.MoveGroupCommander("manipulator")

    rospy.loginfo("Current pose:") #print current end effector pose
    rospy.loginfo(group.get_current_pose())
    
    poses = [
        create_pose(0.4,  0.0, 0.3),   # centre
        create_pose(0.4,  0.1, 0.3),   # right
        create_pose(0.5,  0.1, 0.3),   # forward-right
        create_pose(0.5,  0.0, 0.3),   # forward-centre
        create_pose(0.4,  0.0, 0.3),   # back to start
    ]

    # ---- EXECUTE THE SEQUENCE ONCE ----
    for i in range(len(poses)):
        if rospy.is_shutdown():
            break

        rospy.loginfo(f"Moving to waypoint {i+1}")
        group.set_pose_target(poses[i])

        success = group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

        if not success:
            rospy.logwarn("Planning/Execution failed for this waypoint")
            break

    rospy.loginfo("Sequence complete.")
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
