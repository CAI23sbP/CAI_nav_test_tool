import numpy as np
import rospy

class DWA_RL_RewardCalculator:
    def __init__(
        self,
        robot_radius: float,
        safe_dist: float,
        goal_radius: float,  
    ):
        self.dec_obs = rospy.get_param("/number_decimals_precision_obs_reward", 1)
        self.forwards_reward = rospy.get_param("/forwards_reward",5)
        self.invalid_penalty = rospy.get_param("/xinvalid_penalty",20)
        self.end_episode_points = rospy.get_param("/end_episode_points",2000)
        self.goal_reaching_points = rospy.get_param("/goal_reaching_points",500)
        self.prox_penalty1 = -1 
        self.prox_penalty2 = -3.5
        self.curr_reward = 0
        # additional info will be stored here and be returned alonge with reward.
        self.info = {}
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.last_goal_dist = None
        self.last_dist_to_path = None
        self.last_action = None
        self._curr_dist_to_path = None
        self.safe_dist = safe_dist

    def _reset(self):
        """
        reset variables related to current step
        """
        self.curr_reward = 0
        self.info = {}

    def reset(self):
        """
        reset variables related to the episode
        """
        self.last_goal_dist = None
        self.last_dist_to_path = None
        self.last_action = None
        self._curr_dist_to_path = None

    def get_reward(self,obs_dict, action):
        obs_dict['stacked_obs']
        distance = obs_dict['distance']
        scan = obs_dict['scan']
        self._reset()
        self._reached_goal(distance = distance)
        self._prevent_stopping(action = action)
        self._reward_safe_dist(scan = scan)
        return self.curr_reward, self.info
    
    def _prevent_stopping(self,action):
        if self.last_action is not None:
            if self.last_action != [0,0]:
                self.curr_reward += self.forwards_reward
        self.last_action = action
    
    def _reached_goal(self,distance):
        if distance < self.goal_radius:
            self.curr_reward += self.goal_reaching_points
            self.info["is_done"] = True
            self.info["done_reason"] = 2
            self.info["is_success"] = 1
        else:
            self.info["is_done"] = False

    def _reward_safe_dist(self, scan):
        if scan.min()> self.robot_radius:
            reward = self.prox_penalty1 / scan.round(self.dec_obs)
        else:
            reward = self.prox_penalty2 / scan.round(self.dec_obs)
            self.info["crash"] = True
            self.info["is_done"] = True
            self.info["is_success"] = 0
            self.info["done_reason"] = 1
        self.curr_reward += reward
    
    
    def _not_reached_goal(self):
        return -1*self.end_episode_points