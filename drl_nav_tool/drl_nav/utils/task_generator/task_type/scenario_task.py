from .task_tree import TaskTree
from .base_task import BaseTask
import os,rospy,rospkg,sys
#sys.path.append(os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav/utils/")

@TaskTree.register("ScenarioTask")
class ScenarioTask(BaseTask):
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

        self.obstacles_manager.spawn_human_scenario()           
        robot_init, robot_goal = self.obstacles_manager.spawn_robot_scenario()

        for manager in self.robot_manager:
            manager.reset(start_pos=robot_init , goal_pos=robot_goal )



        return False, (0, 0, 0)
