import rospy
from plant_msgs.msg import PumpState
import serial


def main():
    rospy.init_node('pump_bridge_node')
    rospy.loginfo("Pump bridge node started.")

    pub = rospy.Publisher('/pump/state', PumpState, queue_size=10)
    rate = rospy.Rate(10)

    # open serial connection to Arduino, subject to change 
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    rospy.loginfo("Listening to Arduino on /dev/ttyACM0")


    while not rospy.is_shutdown():
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            rospy.loginfo("Received: %s", line)
            try:
                # Parse "water_level,flow_rate,pump_on" from the Arduino.
                water_level, flow_rate, pump_on = line.split(",")
                rospy.loginfo("Logged - Water Level: %s, Flow Rate: %s, Pump On: %s", water_level, flow_rate, pump_on)
                msg = PumpState()
                msg.header.stamp = rospy.Time.now()
                msg.water_level = int(water_level)
                msg.flow_rate = int(flow_rate)
                msg.pump_on = pump_on.strip().lower() in ("1", "true", "on", "yes")

                pub.publish(msg)
            except Exception as e:
                rospy.logerr("Error parsing serial data: %s", e)
        rate.sleep()

        



if __name__ == '__main__':
   main()
