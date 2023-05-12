import random
import rospy
import sys
from .task_tree import TaskTree
from .base_task import BaseTask
import os
#sys.path.append(os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav/utils/")
from config import Constants

@TaskTree.register("RandomTask")
class RandomTask(BaseTask):
    """
        The random task spawns static and dynamic
        obstacles on every reset and will create
        a new robot start and goal position for
        each task.
    """

    def reset(
        self, start=None, goal=None,
        static_obstacles=None, dynamic_obstacles=None,
    ):
        return super().reset(
            lambda: self._reset_robot_and_obstacle(
                start=start, goal=goal,
                static_obstacles=static_obstacles,
                dynamic_obstacles=dynamic_obstacles,
            )
        )

    def _reset_robot_and_obstacle(
        self, start=None, goal=None, static_obstacles=None,
        dynamic_obstacles=None, 
    ):
        """you must use it , it help code to auto run"""
        robot_positions = []
        obs_init_positions = []
        dynamic_obstacles = random.randint(
            Constants.TaskMode.Random.MIN_DYNAMIC_OBS,
            Constants.TaskMode.Random.MAX_DYNAMIC_OBS
        ) if dynamic_obstacles == None else dynamic_obstacles
        static_obstacles = random.randint(
            Constants.TaskMode.Random.MIN_STATIC_OBS,
            Constants.TaskMode.Random.MAX_STATIC_OBS
        ) if static_obstacles == None else static_obstacles

        init_positions_static = self.obstacles_manager.spawn_static_random(
            static_obstacles=static_obstacles,
            forbidden_zones=robot_positions
        ) 

        init_positions_human = self.obstacles_manager.spawn_human_random(
            dynamic_obstacles=dynamic_obstacles,
            forbidden_zones=robot_positions+init_positions_static
        )

        obs_init_positions=init_positions_human+init_positions_static

        for manager in self.robot_manager:
            manager.reset(forbidden_zones=obs_init_positions)



        return False, (0, 0, 0)
