import rospy
import actionlib

from my_simulations.msg import dispensingAction, dispensingGoal


def feedback_cb(feedback):
    rospy.loginfo(
        "Dispense progress: %f%% | state: %s",
        feedback.percent_complete,
        feedback.state
    )


def done_cb(status, result):
    rospy.loginfo("Dispense finished.")
    rospy.loginfo("Success: %s", result.success)
    rospy.loginfo("Message: %s", result.message)
    rospy.loginfo("Actual volume dispensed: %.2f ml", result.actual_volume)


if __name__ == "__main__":
    rospy.init_node("test_dispense_client")

    client = actionlib.SimpleActionClient("dispense_water", dispensingAction)

    rospy.loginfo("Waiting for dispense action server...")
    client.wait_for_server()

    rospy.loginfo("Server connected.")

    goal = dispensingGoal()

    # TEST PARAMETERS
    goal.plant_id = 1
    goal.target_volume = 10.0  # ml

    rospy.loginfo("Sending goal...")
    client.send_goal(goal,
                     done_cb=done_cb,
                     feedback_cb=feedback_cb)

    client.wait_for_result()

    rospy.loginfo("Client finished.")