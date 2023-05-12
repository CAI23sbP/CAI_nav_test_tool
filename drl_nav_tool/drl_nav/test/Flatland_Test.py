import rospy
import sys
import os 
from simulator_tree import SimulatorTree
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/manager/')
from Obstacle_manager import ObstacleManager
from Robot_manager import RobotManager
from Map_manager import MapManager
from manager.srv import GetDistanceMap
from config import Constants
from task_type.task_tree import TaskTree
from task_type.random_task import RandomTask
from task_type.scenario_task import ScenarioTask

@SimulatorTree.register("TEST")
class FlatlandTest(object):
    def __init__(self,namespace):
        self._namespace = namespace
        self._ns_prefix = "" if namespace == "" else "/" + namespace + "/"
        self._step_size = Constants.STEP_SIZE

    def before_reset_task(self):
        pass

    def after_reset_task(self):
        pass

    def get_predefined_task(self,namespace, mode,  **kwargs):
        """
        Gets the task based on the passed mode
        """
        rospy.wait_for_service("/distance_map")
        service_client_get_map = rospy.ServiceProxy("/distance_map", GetDistanceMap)
        map_response = service_client_get_map()
        map_manager = MapManager(map_response)

        obstacle_manager = ObstacleManager(namespace,map_manager)
        robot_manager = self.create_robot_managers(namespace, map_manager)

        task = TaskTree.instantiate(
            mode,
            obstacle_manager,
            robot_manager,
            namespace=namespace,
            **kwargs
        )
        return task

    def create_robot_managers(self, namespace, map_manager):
        # Read robot setup file
        
        robots = self.create_default_robot_list(
            Constants.RobotManager.NAME,
            Constants.RobotManager.LOCAL_PLANNER.lower(),
            Constants.RobotManager.AGENT_NAME
        )

        robot_managers = []
        for robot in robots:

            # amount = robot["amount"]
            # for r in range(0, amount):

                # name = f"{robot['model']}_{r}_{len(robot_managers)}"  
            name = robot['model']
            robot_managers.append(
                RobotManager(namespace + "/" + name, map_manager, robot)
            )

        
        return robot_managers

    def create_default_robot_list(self,robot_model, planner, agent):
        
        return [{
            "model": robot_model,
            "planner": planner,
            "agent": agent,
            "amount": 1
        }]