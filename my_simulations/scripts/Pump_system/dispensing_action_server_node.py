import time
import rospy
import actionlib

from std_msgs.msg import String
from my_simulations.msg import (
    dispensingAction,
    dispensingFeedback,
    dispensingResult
)

class DispensingActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('dispense_water', dispensingAction, execute_cb=self.execute, auto_start=False)
        self.server.start()
        rospy.loginfo("Dispensing Action Server started.")

        self.pump_command_pub = rospy.Publisher('/pump_command', String, queue_size=10)
        self.pump_rate = 1.417 #ml/s 


    def start_pump(self):
        # Simulate starting the pump for the given plant ID
        rospy.loginfo("Starting pump")
        self.pump_command_pub.publish("START")
        rospy.loginfo("Pump START command sent.")
    
    def stop_pump(self):
        rospy.loginfo("Stopping pump")
        self.pump_command_pub.publish("STOP")
        rospy.loginfo("Pump STOP command sent.")
    
    def execute(self, goal):
        feedback = dispensingFeedback()
        result = dispensingResult()
        plant_id = goal.plant_id
        target_volume = goal.target_volume
        current_volume = 0.0

        rospy.loginfo("Received dispense request for plant = %.2f seconds, target volume = %.2f ml", plant_id, target_volume)

        if target_volume <= 0:
            rospy.logwarn("Invalid dispense request for plant %d: target volume must be positive.", plant_id)
            result.success = False
            result.message = "Invalid parameters: target volume must be positive."
            self.server.set_aborted(result, "Invalid parameters")
            return
        
        start_time = rospy.Time.now().to_sec()
        try: 
            feedback.percent_complete = 0.0
            feedback.state = "starting"
            self.server.publish_feedback(feedback)
            self.start_pump()
            rate = rospy.Rate(10)  # 10 Hz feedback update rate
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    rospy.logwarn("Dispense request for plant %d cancelled.", plant_id)
                    self.stop_pump()
                    result.success = False
                    result.message = "Dispense cancelled"
                    result.actual_volume = 0.0
                    self.server.set_preempted(result, "Dispense cancelled")
                    return
                elapsed_time = rospy.Time.now().to_sec() - start_time
                current_volume = elapsed_time * self.pump_rate
                feedback.percent_complete = min(100.0, (current_volume / target_volume) * 100.0)
                feedback.state = "dispensing"
                self.server.publish_feedback(feedback)

                if current_volume >= target_volume:
                    rospy.loginfo("Target volume reached for plant %d: dispensed %.2f ml in %.2f seconds", plant_id, current_volume, elapsed_time)
                    self.stop_pump()
                    result.success = True
                    result.message = "Dispense completed successfully"
                    result.actual_volume = result.actual_volume = min(current_volume, target_volume)
                    self.server.set_succeeded(result, "Dispense completed")
                    return
                
                rate.sleep()
        except Exception as e:
            rospy.logerr(f"Dispense action failed: {e}")
            self.stop_pump()

            result.success = False
            result.message = str(e)
            result.actual_volume = result.actual_volume = min(current_volume, target_volume)
            self.server.set_aborted(result)


if __name__ == "__main__":
    rospy.init_node("dispense_action_server")
    DispensingActionServer()
    rospy.spin()