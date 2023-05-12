import rospy
from pedsim_msgs.msg import Ped, AgentGroup, AgentGroups
from geometry_msgs.msg import Point
from pedsim_srvs.srv import SpawnPeds
from std_srvs.srv import Trigger
from std_srvs.srv import SetBool
from Base_human import BaseHuman
import sys
import random
import os 
#sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/')
from Human_tree import HumanTree
from config import Constants

T = Constants.TaskMode.WAIT_FOR_SERVICE_TIMEOUT

@HumanTree.register("PEDHUMAN")
class PedHuman(BaseHuman):
    def __init__(self, namespace):
        super().__init__(namespace)
        self._namespace = namespace
        self._ns_prefix = "" if namespace == "" else "/" + namespace + "/"

        rospy.wait_for_service("/pedsim_simulator/spawn_peds", timeout=T)
        rospy.wait_for_service("/pedsim_simulator/reset_all_peds", timeout=T)
        rospy.wait_for_service("/pedsim_simulator/remove_all_peds", timeout=T)

        self._spawn_peds_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}pedsim_simulator/spawn_peds", SpawnPeds
        ) 
        self._reset_peds_srv = rospy.ServiceProxy(
            "/pedsim_simulator/reset_all_peds", Trigger
        )
        self._remove_peds_srv = rospy.ServiceProxy(
            "/pedsim_simulator/remove_all_peds", SetBool
        )
        self._obstacles_amount=0


    def spawn_human_agents(self, dynamic_obstacles):
        """
        maximum number of human determined in obstacle_manager
        """
        if len(dynamic_obstacles) <= 0:
            return
        peds = [PedHuman.create_human_msg(p, i) for i, p in enumerate(dynamic_obstacles)]
        self._obstacles_amount=len(peds)
        spawn_ped_msg = SpawnPeds()
        spawn_ped_msg.peds = peds
        self._spawn_peds_srv(peds)

    def reset_all_human(self):
        self._reset_peds_srv()

    def remove_all_humans(self):
        self._remove_peds_srv(True)

    @staticmethod
    def create_human_msg(ped,id):
        msg = Ped()
        msg.id = id

        pos = Point()
        pos.x = ped[0][0]
        pos.y = ped[0][1]
        msg.pos = pos
        # ped_type = ["adult", "child", "elder","forklift", "servicerobot", "robot" ]
        ped_type = random.choices(["adult", "child", "elder" ], weights=[3,5,4])[0]
        
        if ped_type=="adult":
            ped_max_vel = random.randint(3,14)/10
            force_desired = 1
        elif ped_type == "elder":
            ped_max_vel =  random.randint(3,9)/10
            force_desired = 0.5
        else:
            ped_max_vel = random.randint(3,11)/10
            force_desired = 1
            
        msg.yaml_file = Constants.ObstacleManager.Human.YAML_FILE
        msg.number_of_peds = 1
        msg.type=ped_type
        msg.vmax = ped_max_vel
        msg.start_up_mode = "default"
        msg.wait_time = 0.0
        msg.trigger_zone_radius = 1.0
        msg.chatting_probability = 0.00
        msg.tell_story_probability = 0
        msg.group_talking_probability = 0.00
        msg.talking_and_walking_probability = 0.00
        msg.requesting_service_probability = 0.00
        msg.requesting_guide_probability = 0.00
        msg.requesting_follower_probability = 0.00
        msg.max_talking_distance = 5
        msg.max_servicing_radius = 5
        msg.talking_base_time = 10
        msg.tell_story_base_time = 0
        msg.group_talking_base_time = 10
        msg.talking_and_walking_base_time = 6
        msg.receiving_service_base_time = 20
        msg.requesting_service_base_time = 30
        msg.force_factor_desired = force_desired
        msg.force_factor_obstacle = 2.0
        msg.force_factor_social = 5
        msg.force_factor_robot = 1

        waypoints = []

        for w in ped:
            new_waypoint = Point()
            new_waypoint.x = w[0]
            new_waypoint.y = w[1]

            waypoints.append(new_waypoint)

        msg.waypoints = waypoints

        msg.waypoint_mode = 0

        return msg
