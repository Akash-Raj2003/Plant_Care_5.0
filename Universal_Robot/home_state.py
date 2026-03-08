import rtde_control
import rtde_receive
import time

rtde_c = rtde_control.RTDEControlInterface("169.254.106.99")
rtde_r = rtde_receive.RTDEReceiveInterface("169.254.106.99")
init_q = rtde_r.getActualQ()


home_q = [1.57, -2, 2.5, -0.5, 1.57, 0]



rtde_c.moveJ(home_q, 1.0, 1.0)

rtde_c.moveL(home_q, 1.0, 1.0)

rtde_c.servoStop()
rtde_c.stopScript()

