import rospy
import roslaunch
import os
import yaml
import time
import math
from flatland_msgs.srv import (
    MoveModelRequest, 
    MoveModel, 
    SpawnModel, 
    SpawnModelRequest
)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped ,Pose2D
from std_srvs.srv import Empty

from config import Constants, Test
from flatland_msgs.msg import MoveModelMsg

T = Constants.TaskMode.WAIT_FOR_SERVICE_TIMEOUT
class RobotManager:
    
    PLUGIN_PROPS_TO_EXTEND = {
        "DiffDrive": ["odom_pub", "twist_sub"],
        "Laser": ["topic"] 
    }

    def __init__(self,name_space, map_manager,robot_setup):

        self.map_manager = map_manager
        self.start_pos = [0, 0]
        self.goal_pos = [0, 0]
        self.namespace=name_space
        self.ns_prefix = lambda *topic: os.path.join(self.namespace, *topic)
        self.robot_setup= robot_setup

        self.goal_radius = Test.GOAL_DISTANCE + 1
        self.is_goal_reached = False
        self._move_robot_pub = rospy.Publisher(
            "move_model", MoveModelMsg, queue_size=10
        )
        rospy.wait_for_service("move_model", timeout=T)

        # rospy.wait_for_service(f"{self.namespace}move_model", timeout=T)

        self._spawn_model_srv = rospy.ServiceProxy(
            f"{self.namespace}spawn_model", SpawnModel
        )
        # self._spawn_model_from_string_srv = rospy.ServiceProxy(
        #     f"{self.namespace}spawn_model_from_string", SpawnModel
        # ) 
        self._spawn_model_from_string_srv = rospy.ServiceProxy(
            f"/spawn_model_from_string", SpawnModel
        )
        
        self._move_model_srv = rospy.ServiceProxy(
            f"move_model", MoveModel, persistent=True
        )
        self._clear_costmaps_srv = rospy.ServiceProxy(
            self.ns_prefix(self.namespace, "move_base", "clear_costmaps"), 
            Empty
        )
        self.record_data = Test.RECORDING

        self.position = self.start_pos


    def _robot_name(self):

        return self.namespace
    
    def spawn_robot(self, name, namespace_appendix=None, complexity=1):

        yaml_path = Constants.RobotManager.MODEL_YAML_PATH
        file_content = self._update_plugin_topics(
            self._read_yaml(yaml_path), 
            name
        )
        # rospy.logerr(f"[namespace_appendix]{namespace_appendix}") # burger
        self._spawn_model(
            yaml.dump(file_content), 
            name, 
            os.path.join(self.namespace, namespace_appendix) if len(namespace_appendix) > 0 else self.namespace, 
            [0, 0, 0],
            srv=self._spawn_model_from_string_srv
        )

    def _read_yaml(self, yaml_path):
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)

    def _update_plugin_topics(self, file_content, namespace):

        plugins = file_content["plugins"]

        for plugin in plugins:
            if RobotManager.PLUGIN_PROPS_TO_EXTEND.get(plugin["type"]):
                prop_names = RobotManager.PLUGIN_PROPS_TO_EXTEND.get(plugin["type"])

                for name in prop_names:
                    plugin[name] = os.path.join(namespace, plugin[name])

        return file_content
    
    def _spawn_model(self, yaml_path, name, namespace, position, srv=None):
        request = SpawnModelRequest()
        request.yaml_path = yaml_path
        request.name = name
        request.ns = namespace
        request.pose.x = position[0]
        request.pose.y = position[1]
        request.pose.theta = position[2]

        if srv == None:
            srv = self._spawn_model_srv

        srv(request)

    def set_up_robot(self):
        self.robot_radius = Constants.RobotManager.RADIUS
        self.spawn_robot(self.namespace,"",self._robot_name())
        self.move_base_goal_pub = rospy.Publisher( self.namespace+"/move_base_simple/goal", PoseStamped, queue_size=10)
        self.pub_goal_timer = rospy.Timer(rospy.Duration(0.25), self.publish_goal_periodically)
        self.launch_robot(self.robot_setup)
        rospy.Subscriber(
            os.path.join(self.namespace, "odom"), 
            Odometry, 
            self.robot_pos_callback
        )
        # rospy.logerr(f"[self.namespace] : {self.namespace}") #/burger_0_0

    def reset(self, forbidden_zones=[], start_pos=None, goal_pos=None, move_robot=True):
        self.start_pos, self.goal_pos = self.generate_new_start_and_goal(
            forbidden_zones, start_pos, goal_pos
        )

        rospy.set_param("goal", str(list(self.goal_pos)))
        rospy.set_param("start", str(list(self.start_pos)))

        self.publish_goal(self.goal_pos)

        if move_robot:
            self.move_robot_to_start()

        self.set_is_goal_reached(self.start_pos, self.goal_pos)

        time.sleep(0.1)

        try:
            self._clear_costmaps_srv()
        except:
            pass
        return self.position, self.goal_pos # self.start_pos, self.goal_pos
    
    def launch_robot(self, robot_setup):
        
        roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(
            ["simulation_bringup", "robot.launch"]
        )

        print(f"START WITH MODEL: {robot_setup['model']}" )
        print(f"START WITH MODEL: {robot_setup['planner']}" )
        
        args = [
            f"model:={robot_setup['model']}",
            f"local_planner:={robot_setup['planner']}",
            f"namespace:={self.namespace}",
            f"complexity:={Constants.RobotManager.COMPLEXITY}",
            f"record_data:={self.record_data}",
            *([f"agent_name:={robot_setup.get('agent')}"] if robot_setup.get('agent') else [])
        ]

        self.process = roslaunch.parent.ROSLaunchParent(
            roslaunch.rlutil.get_or_generate_uuid(None, False),
            [(*roslaunch_file, args)]
        )
        self.process.start()

        # Overwrite default move base params
        base_frame = rospy.get_param(os.path.join(self.namespace, "robot_base_frame"))
        sensor_frame = rospy.get_param(os.path.join(self.namespace, "robot_sensor_frame"))
        map_frame = rospy.get_param(os.path.join(self.namespace, "map_frame"), "map")
        odom_frame = rospy.get_param(os.path.join(self.namespace, "odom_frame"), "odom")

        rospy.set_param(
            os.path.join(self.namespace, "move_base", "global_costmap", "robot_base_frame"),
            self.namespace.replace("/", "") + "/"+ base_frame
        )
        rospy.set_param(
            os.path.join(self.namespace, "move_base", "local_costmap", "robot_base_frame"),
            self.namespace.replace("/", "") + "/"+  base_frame
        )

        rospy.set_param(
            os.path.join(self.namespace, "move_base", "local_costmap", "sensor_frame"),
            self.namespace.replace("/", "") +"/"+ sensor_frame
        )
        rospy.set_param(
            os.path.join(self.namespace, "move_base", "global_costmap", "sensor_frame"),
            self.namespace.replace("/", "") + "/"+ base_frame
        )

        rospy.set_param(
            os.path.join(self.namespace, "move_base", "local_costmap", "global_frame"),
            self.namespace.replace("/", "") +"/"+ odom_frame
        )
        rospy.set_param(
            os.path.join(self.namespace, "move_base", "global_costmap", "global_frame"),
            map_frame
        )

    def publish_goal_periodically(self, _):
        if self.goal_pos != None:
            self.publish_goal(self.goal_pos)

    def generate_new_start_and_goal(self, forbidden_zones, start_pos, goal_pos):
        new_start_pos = self._default_position(
            start_pos,
            self.map_manager.get_random_pos_on_map(
                self.robot_radius + Constants.RobotManager.SPAWN_ROBOT_SAFE_DIST,
                forbidden_zones
            )
        )

        new_goal_pos = self._default_position(
            goal_pos,
            self.map_manager.get_random_pos_on_map(
                self.robot_radius + Constants.RobotManager.SPAWN_ROBOT_SAFE_DIST,
                [
                    *forbidden_zones,
                    (
                        new_start_pos[0], 
                        new_start_pos[1], 
                        self.goal_radius
                    )
                ]
            )
        )

        return new_start_pos, new_goal_pos

    def publish_goal(self, goal):
        goal_msg = PoseStamped()
        goal_msg.header.seq = 0
        goal_msg.header.stamp = rospy.get_rostime()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = goal[0]
        goal_msg.pose.position.y = goal[1]

        goal_msg.pose.orientation.w = 0
        goal_msg.pose.orientation.x = 0
        goal_msg.pose.orientation.y = 0
        goal_msg.pose.orientation.z = 1

        self.move_base_goal_pub.publish(goal_msg)

    def move_robot_to_start(self):
        if not self.start_pos == None:

            self.move_robot_to_pos(self.start_pos)

    def move_robot_to_pos(self, pos):
        self.move_robot(pos, name=self.namespace)

    def move_robot(self, pos, name=None):
        pose = Pose2D()
        pose.x = pos[0]
        pose.y = pos[1]
        pose.theta = pos[2]

        move_model_request = MoveModelRequest()
        move_model_request.name = name 
        move_model_request.pose = pose
        self._move_model_srv(move_model_request)

    def _default_position(self, pos, callback_pos):
        if not pos == None:
            return pos

        return callback_pos

    def robot_pos_callback(self, data):
        current_position = data.pose.pose.position

        self.position = [current_position.x, current_position.y]

        self.set_is_goal_reached(
            self.position,
            self.goal_pos
        )

    def set_is_goal_reached(self, start, goal):
        distance_to_goal = math.sqrt(
            (start[0] - goal[0]) ** 2
            + (start[1] - goal[1]) ** 2 
        )

        self.is_goal_reached = distance_to_goal < self.goal_radius

    def is_done(self):
        return self.is_goal_reached
    

