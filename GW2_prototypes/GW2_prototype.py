#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import serial

from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import PlanningScene, ObjectColor

PORT = "/dev/ttyUSB0"
BAUD = 9600


def main():
    # ---- INIT ROS + MOVEIT ----
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("ur5_GW2_prototype", anonymous=True)

    group = moveit_commander.MoveGroupCommander("manipulator")
    scene = PlanningSceneInterface()

    rospy.sleep(2.0)  # waits to connect

    planning_frame = group.get_planning_frame()

    # add plant
    plant_pose = PoseStamped()
    plant_pose.header.frame_id = planning_frame
    plant_pose.pose.position.x = 0.11
    plant_pose.pose.position.y = -0.79
    plant_pose.pose.position.z = 0.23

    rospy.loginfo("Adding plant")
    scene.add_cylinder("plant", plant_pose, height=0.55, radius=0.25)

    # add soil
    soil_pose = PoseStamped()
    soil_pose.header.frame_id = planning_frame
    soil_pose.pose.position.x = 0.11
    soil_pose.pose.position.y = -0.79
    soil_pose.pose.position.z = 0.23

    rospy.loginfo("Adding soil")
    scene.add_cylinder("soil", soil_pose, height=0.58, radius=0.22)

    rospy.sleep(1.5)  # RViz update time

    # colour in
    scene_pub = rospy.Publisher("/planning_scene", PlanningScene, queue_size=10)

    # Plant (green)
    color_plant = ObjectColor()
    color_plant.id = "plant"  
    color_plant.color.r = 0.0
    color_plant.color.g = 1.0
    color_plant.color.b = 0.0
    color_plant.color.a = 1.0

    # Soil (brown)
    color_soil = ObjectColor()
    color_soil.id = "soil"
    color_soil.color.r = 0.59   # brown
    color_soil.color.g = 0.29
    color_soil.color.b = 0.0
    color_soil.color.a = 1.0

    # make planning scene and change colours
    scene_msg = PlanningScene()
    scene_msg.is_diff = True
    scene_msg.object_colors.append(color_plant)
    scene_msg.object_colors.append(color_soil)

    # Publish several times so RViz receives it
    for _ in range(8):
        scene_pub.publish(scene_msg)
        rospy.sleep(0.1)

    # ---- OPEN SERIAL PORT ----
    try:
        ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.1)
        rospy.loginfo(f"Opened serial port {PORT} at {BAUD} baud")
    except Exception as e:
        rospy.logerr(f"Failed to open serial port: {e}")
        ser = None

    # ---- MAIN LOOP ----
    try:
        while not rospy.is_shutdown():

            line = ""
            if ser is not None:
                line = ser.readline().decode(errors="ignore").strip()

            if line:
                rospy.loginfo(f"[Serial] Received: {line}")

                if line.lower() == "water":
                    rospy.loginfo("Moving to watering pose")

                    joints = group.get_current_joint_values()
                    joints[0] = -4.312
                    joints[1] = -1.969
                    joints[2] = -0.433
                    joints[3] = -1.602
                    joints[4] = 1.531
                    joints[5] = -5.689
                    
                    rospy.loginfo(f"Moving to: {joints}")
                    success = group.go(joints, wait=True)  # <-- FIXED
                    group.stop()
                 
                    rospy.loginfo(f"Move success: {success}")
                    ser.write("Done\n".encode())


            rospy.sleep(0.01)

    except rospy.ROSInterruptException:
        pass

    finally:
        rospy.loginfo("Shutting down node.")
        if ser is not None:
            ser.close()
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()

