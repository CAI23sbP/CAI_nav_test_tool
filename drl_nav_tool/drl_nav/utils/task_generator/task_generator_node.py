#! /usr/bin/env python3
import rospy
from std_msgs.msg import Int16, Empty as EmptyMsg
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
import sys
from sensor_msgs.msg import LaserScan
import numpy as np 
import os 
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/task_generator/')
from simulator_tree import SimulatorTree
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/')
from config import Test
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/test/')
from Flatland_Test import FlatlandTest

class TaskGenerator:
    """
    Task Generator Node
    Will initialize and reset all tasks. The task to use is read from the `/task_mode` param.
    """

    def __init__(self) -> None:
        ## Params
        self.task_mode = Test.TASK_MODE
        self.sim_type = "TEST"
        self.collision_flag = False

        ## Publishers
        self.pub_scenario_reset = rospy.Publisher("scenario_reset", Int16, queue_size=1)
        self.pub_scenario_finished = rospy.Publisher('scenario_finished', EmptyMsg, queue_size=10)
        self.scan = rospy.Subscriber('/burger/scan',LaserScan, self.is_collision)
        ## Services
        rospy.Service("reset_task", Empty, self.reset_task_srv_callback)


        rospy.loginfo(f"Launching task mode: {self.task_mode}")

        self.start_time = rospy.get_time()
        self.env = SimulatorTree.instantiate(self.sim_type)("")

        self.task = self.env.get_predefined_task("", self.task_mode)

        self.number_of_resets = 0
        self.desired_resets = Test.DESIRED_RESET

        self.srv_start_model_visualization = rospy.ServiceProxy("start_model_visualization", Empty)
        self.srv_start_model_visualization(EmptyRequest())

        self.reset_task()

        rospy.sleep(2)

        try:
            rospy.set_param("task_generator_setup_finished", True)
            self.srv_setup_finished = rospy.ServiceProxy("task_generator_setup_finished", Empty)
            self.srv_setup_finished(EmptyRequest())
        except:
            pass

        self.number_of_resets = 0

        self.reset_task()

        ## Timers
        rospy.Timer(rospy.Duration(0.5), self.check_task_status)
        
    def is_collision(self,msg):
        scan_array = np.asarray(msg.ranges)
        d_min = np.nanmin(scan_array)
    
        if d_min > 0.5:
            self.collision_flag = False
        if d_min <= 0.35 and not self.collision_flag:
            self.collision_flag = True
           

    def check_task_status(self, _):
        if self.task.is_done():
            self.reset_task()

        if self.collision_flag:
            self.reset_task()


    def reset_task(self):
        self.start_time = rospy.get_time()
        self.number_of_resets += 1

        self.env.before_reset_task()

        is_end = self.task.reset()

        self.pub_scenario_reset.publish(self.number_of_resets)
        self._send_end_message_on_end(is_end)

        self.env.after_reset_task()

        rospy.loginfo("=============")
        rospy.loginfo("Task Reseted!")
        rospy.loginfo("=============")


    def reset_task_srv_callback(self, req):
        rospy.logdebug("Task Generator received task-reset request!")

        self.reset_task()

        return EmptyResponse()

    def _send_end_message_on_end(self, is_end):
        if (
            (not is_end ) 
            or ( self.number_of_resets < self.desired_resets)
        ):
            return

        rospy.loginfo("Shutting down. All tasks completed")

        # Send Task finished to Backend
        while self.pub_scenario_finished.get_num_connections() <= 0:
            pass

        self.pub_scenario_finished.publish(EmptyMsg())

        rospy.signal_shutdown("Finished all episodes of the current scenario")


if __name__ == "__main__":
    rospy.init_node("task_generator")

    task_generator = TaskGenerator()

    rospy.spin()
