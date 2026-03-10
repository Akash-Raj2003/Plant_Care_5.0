#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from plant_msgs.msg import PlantMoisture

import serial


def main():
    rospy.init_node('plant_radio_bridge_node')
    rospy.loginfo("Plant radio bridge node started.")


    # open serial connection to Arduino
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    rospy.loginfo("Listening to Arduino on /dev/ttyACM0")

    #create publisher for plant moisture alerts
    pub = rospy.Publisher("/plant/moisture_alert", PlantMoisture, queue_size=10)

    while not rospy.is_shutdown():
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            rospy.loginfo("Received: %s", line)
            
            try:
                #split the recevied message by commas '.'
                plant_id, moisture, low = line.split(",")
                rospy.loginfo("Logged - Plant ID: %s, Moisture: %s, Low: %s", plant_id, moisture, low)

                msg = PlantMoisture()
                msg.plant_id = int(plant_id)
                msg.moisture_level = float(moisture)
                msg.low_moisture = bool(int(low))

                pub.publish(msg)

            except Exception as e:
                rospy.logwarn("Failed to log line '%s': %s", line, str(e))

if __name__ == '__main__':
   main()
