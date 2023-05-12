import os, sys, rospy, random, yaml, math
import numpy as np
from Base_static import BaseStatic
from Static_tree import StaticTree
#sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/')
from config import Constants
from flatland_msgs.srv import (
    DeleteModelRequest,
    SpawnModel, 
    DeleteModel,
    SpawnModelRequest
)

T = Constants.TaskMode.WAIT_FOR_SERVICE_TIMEOUT

@StaticTree.register("FLATLANDSTATIC")
class FlatlandStaticObs(BaseStatic):
    def __init__(self, namespace):
        super().__init__(namespace)
        self._namespace = namespace
        self._ns_prefix = "" if namespace == "" else "/" + namespace + "/"
        
        rospy.wait_for_service(f"{self._ns_prefix}delete_model", timeout=T)
        
        self._delete_static_obstacles_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}delete_model", DeleteModel
        )
        self._spawn_static_obstacles_from_string_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}spawn_model_from_string", SpawnModel
        )
        self._spawn_static_obstacles_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}spawn_model", SpawnModel
        ) 
        self._obstacles_amount = 0

    def reset_all_static(self):
        pass

    def remove_all_static_obstacles(self):
        for obs in range(self._obstacles_amount):
            obs_name = FlatlandStaticObs.create_obs_name(obs)
            self._delete_static_obstacles(obs_name)
        self._obstacles_amount = 0

    def _delete_static_obstacles(self, name):
        delete_static_obstacles_request = DeleteModelRequest()
        delete_static_obstacles_request.name = name

        self._delete_static_obstacles_srv(delete_static_obstacles_request)

    def spawn_static_obstacles(self, **args):
        self._spawn_random_obstacle(**args )

    def _spawn_random_obstacle(
            self,  position=[0, 0, 0], **args
        ):
        static_obstacles = self._generate_random_static_obstacle(**args)

        obstacle_name = FlatlandStaticObs.create_obs_name(
            self._obstacles_amount
        )
        self._spawn_static_obstacles(
            yaml.dump(static_obstacles), 
            obstacle_name, 
            self._namespace, 
            position, 
            srv=self._spawn_static_obstacles_from_string_srv
        )
        self._obstacles_amount += 1 


    def _spawn_static_obstacles(self, yaml_path, name, namespace, position, srv=None):
        request = SpawnModelRequest()
        request.yaml_path = yaml_path
        request.name = name
        request.ns = namespace
        request.pose.x = position[0]
        request.pose.y = position[1]
        request.pose.theta = position[2]
        if srv == None:
            srv = self._spawn_static_obstacles_srv

        srv(request)

    ## HELPER FUNCTIONS TO CREATE MODEL.YAML
    def _generate_random_static_obstacle(
            self, 
            min_radius=Constants.ObstacleManager.Static.MIN_RADIUS, 
            max_radius=Constants.ObstacleManager.Static.MAX_RADIUS,
            
        ):
        """
            Creates a dict in the flatland model schema.
            Since a lot of the variables are untouched
            the majority of the dict is filled up with
            constants defined in the `Constants` file.
        """
        body = {
            **Constants.ObstacleManager.Static.BODY,
            "type":"static"
        }

        footprint = {
            **Constants.ObstacleManager.Static.FOOTPRINT,
            **self._generate_random_footprint_type(min_radius, max_radius)
        }

        body["footprints"] = [footprint]

        model = {'bodies': [body], "plugins": []}

        return model

    def _generate_random_footprint_type(self, min_radius, max_radius):

        type = "polygon"

        points_amount = random.randint(3, 8) # Defined in flatland definition
        angle_interval = 2 * np.pi / points_amount

        points = []

        for p in range(points_amount):
            angle = random.uniform(0, angle_interval)
            radius = random.uniform(min_radius, max_radius)

            real_angle = angle_interval * p + angle

            points.append([
                math.cos(real_angle) * radius, 
                math.sin(real_angle) * radius
            ])

        return {
            "type": type,
            "points": list(points)
        }


    @staticmethod
    def create_obs_name(number):
        return "obs_static_" + str(number)
