from config import Constants
import sys
import rospy
import json
import os
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/robohumstamap/human_type/')
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/robohumstamap/static_obs_type/')
from Human_tree import HumanTree
from Ped_human import PedHuman
from Random_human import FlatlandRandomHuman
from Static_tree import StaticTree
from Flatland_static_obs import FlatlandStaticObs
from Ped_static_obs import PedStaticObs

import random

class ObstacleManager:
    def __init__(self, namespace,map_manager):
        self.map_manager = map_manager
        
        ## setting config
        self.human_config = Constants.ObstacleManager.Human
        self.human_type = HumanTree.instantiate(self.human_config.HUMAN_TYPE.upper())(namespace)
        self.human_max_radius=self.human_config.HUMAN_MAX_RADIUS

        self.static_config=Constants.ObstacleManager.Static
        self.static_type = StaticTree.instantiate(self.static_config.STATIC_TYPE.upper())(namespace)
        self.static_max_radius=self.static_config.OBSTACLE_MAX_RADIUS

        self.scenario_config = Constants.TaskMode.Scenario
        self.scenario_file = self.scenario_config.YAML_PATH

    def spawn_human_random(self,dynamic_obstacles,forbidden_zones):
        self.human_type.reset_all_human()
        self.human_type.remove_all_humans()
        if Constants.ObstacleManager.IS_HUMAN: 
            rospy.loginfo(f"number of human : {dynamic_obstacles} ")

            if self.human_config.HUMAN_TYPE=="PEDHUMAN":

                for _ in range(dynamic_obstacles):
                    waypoints = []
                    for __ in range(random.choices(range(1, 5), weights=[1,1,2,2])[0]): ## generate random waypoint_number
                        position = self.map_manager.get_random_pos_on_map(safe_dist=self.human_max_radius,forbidden_zones=forbidden_zones)
                        waypoints.append(position)
                    position_list=self.concat_waypoint(waypoints)
                    self.human_type.spawn_human_agents([position_list] )
                return position_list
            
            else:
                position_list=[]
                for _ in range(dynamic_obstacles):
                    position = self.map_manager.get_random_pos_on_map(safe_dist=self.human_max_radius,forbidden_zones=forbidden_zones)
                    self.human_type.spawn_human_agents(position=position )
                    position_list.append(position)
                return position_list
        else:
            return []
        
    def spawn_static_random(self,static_obstacles,forbidden_zones):
        self.static_type.reset_all_static()
        self.static_type.remove_all_static_obstacles()
        

        if Constants.ObstacleManager.IS_STATIC:
            rospy.loginfo(f"number of static : {static_obstacles} ")
            if self.static_config.STATIC_TYPE=="PEDSTATIC":
                save_all_obs=[]
                for _ in range(static_obstacles):
                    position = self.map_manager.get_random_pos_on_map(safe_dist=self.static_max_radius,forbidden_zones=forbidden_zones)
                    save_all_obs.append(position)
                self.static_type.spawn_static_obstacles([save_all_obs])
                return save_all_obs
            else:
                position_list=[]
                for _ in range(static_obstacles):
                    position = self.map_manager.get_random_pos_on_map(
                        safe_dist=self.static_max_radius, 
                        forbidden_zones=forbidden_zones
                    )
                    self.static_type.spawn_static_obstacles(position=position)
                    position_list.append(position)
                return position_list
        else:
            return []

    def concat_waypoint(self,waypoints):
        position_list=[]
        for i in range(len(waypoints)):
            position_list.append([waypoints[i][0],waypoints[i][1],0.7])
        return position_list

    def check_map_scenario(self):
        with open(self.scenario_file) as f:
            scen_data= json.load(f)

        if Constants.MapManager.MAP_PATH + Constants.MapManager.MAP_TYPE != scen_data["map_path"]:
            rospy.logerr("Map path of scenario and map are not the same. Shutting down.")
            rospy.logerr(f"Scenario Map Path {scen_data['map_path']}")
            rospy.logerr(f" Map Path {Constants.MapManager.MAP_PATH + Constants.MapManager.MAP_TYPE}")
            rospy.signal_shutdown("Map path of scenario and map are not the same.")
            sys.exit()


    def spawn_human_scenario(self):
        self.human_type.reset_all_human()
        self.human_type.remove_all_humans()

        if Constants.ObstacleManager.IS_HUMAN: 
            with open(self.scenario_file) as f:
                scen_data= json.load(f)
            dynamic_obstacles = len(scen_data["pedsim_agents"])
            rospy.loginfo(f"number of human : {dynamic_obstacles} ")

            for i in range(dynamic_obstacles):
                self.human_type.spawn_human_agents([[scen_data["pedsim_agents"][i]["pos"]]+scen_data["pedsim_agents"][i]["waypoints"]])
        
    def spawn_robot_scenario(self):
            with open(self.scenario_file) as f:
                scen_data= json.load(f)
            return scen_data["robot_position"], scen_data["robot_goal"]
