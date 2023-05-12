import os, sys, rospy,random
from Base_static import BaseStatic
from Static_tree import StaticTree

from pedsim_msgs.msg import InteractiveObstacle

from geometry_msgs.msg import Pose
from pedsim_srvs.srv import SpawnInteractiveObstacles
from std_srvs.srv import Trigger

#sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/')
from config import Constants

T = Constants.TaskMode.WAIT_FOR_SERVICE_TIMEOUT

@StaticTree.register("PEDSTATIC")
class PedStaticObs(BaseStatic):
    def __init__(self, namespace):
        super().__init__(namespace)
        self._namespace = namespace
        self._ns_prefix = "" if namespace == "" else "/" + namespace + "/"
        
        rospy.wait_for_service("/pedsim_simulator/spawn_interactive_obstacles", timeout=T)
        rospy.wait_for_service("/pedsim_simulator/respawn_interactive_obstacles", timeout=T)
        rospy.wait_for_service("/pedsim_simulator/remove_all_interactive_obstacles", timeout=T)

        self._spawn_peds_srv = rospy.ServiceProxy(
            f"{self._ns_prefix}pedsim_simulator/spawn_interactive_obstacles", SpawnInteractiveObstacles
        ) 
        self._reset_peds_srv = rospy.ServiceProxy(
            "/pedsim_simulator/respawn_interactive_obstacles", SpawnInteractiveObstacles
        )
        self._remove_peds_srv = rospy.ServiceProxy(
            "/pedsim_simulator/remove_all_interactive_obstacles", Trigger
        )
        self._obstacles_amount=0

    def _delete_static_obstacles(self):
        pass
    
    def spawn_static_obstacles(self, position):
        """
        maximum number of human determined in obstacle_manager
        """
        if len(position) <= 0:
            return
        obstacles = [PedStaticObs.create_static_msg(p, i) for i, p in enumerate(position[0])]
        self._obstacles_amount=len(obstacles)
        spawn_ped_msg = SpawnInteractiveObstacles()
        spawn_ped_msg.obstacles=obstacles
        self._spawn_peds_srv(obstacles)

    def reset_all_static(self):
        self._reset_peds_srv()

    def remove_all_static_obstacles(self):
        self._remove_peds_srv()



    @staticmethod
    def create_static_msg(ped,id):
        file_list=[]
        for filename in os.listdir(Constants.ObstacleManager.Static.YAML_FILE):
            file_list.append(filename)
        ## 'shelf', 'long_shelf.model', chair.model'
        msg = InteractiveObstacle()
        msg.yaml_path = Constants.ObstacleManager.Static.YAML_FILE+file_list[random.choices([0,1,2],weights=[3,2,1])[0]]
        msg.name = "static_"+str(id)
        msg.interaction_radius = 4.0
        pos = Pose()
        pos.position.x = ped[0]
        pos.position.y = ped[1]
        msg.type = "static"
        msg.pose = pos

        return msg
