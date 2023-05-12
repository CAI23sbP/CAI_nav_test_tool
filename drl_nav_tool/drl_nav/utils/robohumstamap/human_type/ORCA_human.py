#!/usr/bin/env python3
###TODO check is it launch or not
import random
import numpy as np
import rospy
from geometry_msgs.msg import  Pose2D
import rvo2
from .Human_tree import HumanTree
from .Base_human import BaseHuman
from utils.config import Constants
from flatland_msgs.srv import (   
    DeleteModelRequest,
    SpawnModel,
    SpawnModelRequest,
    DeleteModel,
    DeleteModelRequest
)
# from flatland_msgs.msg import MoveModelMsg

from geometry_msgs.msg import Twist

T = Constants.TaskMode.WAIT_FOR_SERVICE_TIMEOUT
@HumanTree.register("ORCAHUMAN")
class ORCA_Human(BaseHuman):

    """
    ORCA HUMAN 
    docs : https://gamma.cs.unc.edu/RVO2/documentation/2.0/class_r_v_o_1_1_r_v_o_simulator.html
    """

    """
    TODO This human are not conpleted!!!!!!!!!  -> we must generate .cpp about move_model
    """
    def __init__(self, namespace):
        super().__init__(namespace)
        self._namespace = namespace
        self._ns_prefix = "" if namespace == "" else "/" + namespace + "/"
        self.time_step = Constants.STEP_SIZE
        self.params = Constants.ObstacleManager.Human.PARAMS
        self.v_pref=Constants.ObstacleManager.Human.V_PREF
        self.sim_orca = rvo2.PyRVOSimulator(self.time_step, *self.params, 0.3, 1)
        """
        timeStep
        neighborDist
        maxNeighbors
        timeHorizon
        timeHorizonObst
        radius
        maxSpeed
        velocity
        """
        rospy.wait_for_service(f"{self._ns_prefix}spawn_model", timeout=T)
        rospy.wait_for_service(f"{self._ns_prefix}delete_model", timeout=T)
        rospy.wait_for_service(f"{self._ns_prefix}move_model", timeout=T)

        self._spawn_model_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}spawn_model", SpawnModel
        )
        self._spawn_model_from_string_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}spawn_model_from_string", SpawnModel
        )
        self._delete_human_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}delete_model", DeleteModel
        )
        self.agent_action_pubs=list()

    def spawn_human_agents(self, dynamic_obstacles):
        """
        maximum number of human determined in obstacle_manager
        """
        if len(dynamic_obstacles) <= 0:
            return
        
        orca = [self.create_ped_msg(p, i) for i, p in enumerate(dynamic_obstacles)]
        self.goal=[i[1] for i in dynamic_obstacles]
        for agent_no in range(dynamic_obstacles):
            self.agent_action_pubs.append(rospy.Publisher(
                    f"/{agent_no}/cmd_vel_pub", Twist, queue_size=1
                )) 
            
        self._spawn_model_srv(orca)


    def remove_all_humans(self):
        for obs in range(self.sim_orca.getNumAgents()):
            obs_name = ORCA_Human.create_obs_name(obs)

        self._delete_human(obs_name)


    def _delete_human(self, name):
        delete_model_request = DeleteModelRequest()
        delete_model_request.name = name
        self._delete_human_srv(delete_model_request)

    @staticmethod
    def create_obs_name(number):
        return "obs_" + str(number)

    def create_human_msg(self,position,id):
        Max=Constants.ObstacleManager.Human.HUMAN_MAX_RADIUS*10
        radius= random.randrange(1,Max)/10
        self.sim_orca.addAgent(position[0][0],position[0][1],*self.params,radius)
        return self.sim_orca
    
    def step_orca(self):
        pose = Pose2D()
        for agent_no in range(self.sim_orca.getNumAgents()):
            px,py=self.sim_orca.getAgentPosition(agent_no+1)
            pose.x = px
            pose.y = py
            pose.theta = 0
            velocity=np.array(self.goal[agent_no][0]-px,self.goal[agent_no][1]-py)
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity
            self.sim_orca.setAgentPrefVelocity(agent_no+1, tuple(pref_vel))
        #     self._move_robot_pub = rospy.Publisher(
        #     agent_no + "move_model", MoveModelMsg, queue_size=10
        # )
            vx,vy=self.sim_orca.getAgentVelocity(agent_no)
            action_msg = Twist()
            action_msg.linear.x = vx
            action_msg.linear.y = vy
            action_msg.linear.z = 0
            self.agent_action_pubs[agent_no].publish(action_msg)

        self.sim_orca.doStep()