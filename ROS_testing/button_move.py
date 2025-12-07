#!/usr/bin/env python3

import sys
import random

import rospy
import moveit_commander
import serial


PORT = "/dev/ttyACM0"   # change if needed
BAUD = 9600


def main():
    # ---- INIT ROS + MOVEIT ----
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("ur5_button_move", anonymous=True)

    group = moveit_commander.MoveGroupCommander("manipulator")

    # ---- OPEN SERIAL ----
    ser = serial.Serial(PORT, baudrate=BAUD, timeout=0.1)
    rospy.loginfo(f"Opened serial port {PORT} at {BAUD} baud")

    rospy.loginfo("Ready: press Arduino button to move the UR5.")

    try:
        while not rospy.is_shutdown():
            line = ser.readline().decode(errors="ignore").strip()

            if line:
                rospy.loginfo(f"[Serial] {line}")

                if line.lower() == "water":
                    rospy.loginfo("Button press detected → triggering UR5 move...")

                    # ---- INLINE RANDOM JOINT MOVE ----
                    joints = group.get_current_joint_values()
                    for i in range(len(joints)):
                        joints[i] += random.uniform(-0.3, 0.3)
                    rospy.loginfo(f"Moving to: {joints}")
                    group.go(joints, wait=True)
                    group.stop()
                    # -----------------------------------

            rospy.sleep(0.01)

    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.loginfo("Shutting down, closing serial.")
        ser.close()
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()

