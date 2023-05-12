import os, sys, rospy, random, yaml, math
import numpy as np
from Base_human import BaseHuman
#sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/')
from Human_tree import HumanTree
from config import Constants
from flatland_msgs.srv import (
    DeleteModelRequest,
    SpawnModel, 
    DeleteModel,
    SpawnModelRequest,
)

T = Constants.TaskMode.WAIT_FOR_SERVICE_TIMEOUT

@HumanTree.register("FLATLANDRANDOMHUMAN")
class FlatlandRandomHuman(BaseHuman):
    def __init__(self, namespace):
        super().__init__(namespace)
        self._namespace = namespace
        self._ns_prefix = "" if namespace == "" else "/" + namespace + "/"
        
        rospy.wait_for_service(f"{self._ns_prefix}delete_model", timeout=T)

        self._delete_human_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}delete_model", DeleteModel
        )
 
        self._spawn_human_from_string_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}spawn_model_from_string", SpawnModel
        )
        self._spawn_human_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}spawn_model", SpawnModel
        ) 
        self._obstacles_amount = 0
    
    def reset_all_human(self):
        pass

    def remove_all_humans(self):
        for obs in range(self._obstacles_amount):
            obs_name = FlatlandRandomHuman.create_obs_name(obs)

            self._delete_human(obs_name)

        self._obstacles_amount = 0

    def _delete_human(self, name):
        delete_human_request = DeleteModelRequest()
        delete_human_request.name = name

        self._delete_human_srv(delete_human_request)

    def spawn_human_agents(self, **args):
        self._spawn_random_obstacle(**args)

    def _spawn_random_obstacle(
            self, position=[0, 0, 0], **args
        ):

        human = self._generate_random_obstacle( **args)

        obstacle_name = FlatlandRandomHuman.create_obs_name(
            self._obstacles_amount
        )

        self._spawn_human(
            yaml.dump(human), 
            obstacle_name, 
            self._namespace, 
            position, 
            srv=self._spawn_human_from_string_srv
        )
        self._obstacles_amount += 1 

    def _spawn_human(self, yaml_path, name, namespace, position, srv=None):
        request = SpawnModelRequest()
        request.yaml_path = yaml_path
        request.name = name
        request.ns = namespace
        request.pose.x = position[0]
        request.pose.y = position[1]
        request.pose.theta = position[2]

        if srv == None:
            srv = self._spawn_human_srv

        srv(request)

    def _generate_random_obstacle(
            self, 
            min_radius=Constants.ObstacleManager.Human.MIN_RADIUS, 
            max_radius=Constants.ObstacleManager.Human.MAX_RADIUS,
            linear_vel=Constants.ObstacleManager.Human.LINEAR_VEL,
            angular_vel_max=Constants.ObstacleManager.Human.ANGLUAR_VEL_MAX
        ):
   
        body = {
            **Constants.ObstacleManager.Human.BODY,
            "type": "dynamic" 
        }

        footprint = {
            **Constants.ObstacleManager.Human.FOOTPRINT,
            **self._generate_random_footprint_type(min_radius, max_radius)
        }

        body["footprints"] = [footprint]

        model = {'bodies': [body], "plugins": []}

        
        model['plugins'].append({
            **Constants.ObstacleManager.Human.RANDOM_MOVE_PLUGIN,
            'linear_velocity': random.uniform(0, linear_vel),
            'angular_velocity_max': random.uniform(0.2, angular_vel_max)
        })

        return model

    def _generate_random_footprint_type(self, min_radius, max_radius):

        type = "circle"
        radius = random.uniform(min_radius, max_radius)
        return {
            "type": type,
            "radius": radius
        }

    @staticmethod
    def create_obs_name(number):
        return "obs_" + str(number)
