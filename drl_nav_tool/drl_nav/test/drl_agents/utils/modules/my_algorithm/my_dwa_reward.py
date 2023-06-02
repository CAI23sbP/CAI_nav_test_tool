import numpy as np

class MYRewardCalculator:
    def __init__(
        self,
        robot_radius: float,
        safe_dist: float,
        goal_radius: float,  
    ):
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

    def _cal_dist_to_human(self,human_pose):
        dist_to_human = []
        for i in range(len(human_pose)):
 
                dist_to_human.append(np.sqrt(human_pose[i][0]**2+human_pose[i][1]**2))
            
        return np.array(dist_to_human)


    def get_reward(self,obs_dict):
        self._reset()
        
        dist_to_human =self._cal_dist_to_human(obs_dict["human_pose"]) 
        robot_node = obs_dict["robot_node"]

        self._cal_reward_rule(dist_to_human= dist_to_human, goal_in_robot_frame = robot_node)
        return self.curr_reward, self.info
    
        
    def _cal_reward_rule(self,dist_to_human,goal_in_robot_frame):
        """
        reward_goal_reached must be in before reward_collision
        because if you are located reward_collision before reward_goal_reached, then is_done never changed by collision reason 
        """
        self._reward_goal_reached(goal_in_robot_frame, reward=10)
        self._reward_collision(goal_in_robot_frame,dist_to_human, punishment=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=2, penalty_factor=2
        )
        self._reward_safe_dist(dist_to_human, punishment=0.1)

    def _reward_goal_reached(
        self,
        goal_in_robot_frame,
        reward: float = 15,
        *args,
        **kwargs,
    ):
        """
        Reward for reaching the goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward (float, optional): reward amount for reaching. defaults to 15
        """
        if goal_in_robot_frame[0] < self.goal_radius:
            self.curr_reward += reward
            self.info["is_done"] = True
            self.info["done_reason"] = 2
            self.info["is_success"] = 1
        else:
            self.info["is_done"] = False

    def _reward_goal_approached(
        self,
        goal_in_robot_frame,
        reward_factor: float = 0.3,
        penalty_factor: float = 0.5,
        *args,
        **kwargs,
    ):
        """
        Reward for approaching the goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward_factor (float, optional): positive factor for approaching goal. defaults to 0.3
        :param penalty_factor (float, optional): negative factor for withdrawing from goal. defaults to 0.5
        """
        max_dist = np.sqrt(goal_in_robot_frame[5]**2 + goal_in_robot_frame[6]**2)
        dist = goal_in_robot_frame[1] 
        if self.last_goal_dist is not None:
            if (self.last_goal_dist - dist) > 0:
                w = reward_factor 
            else:
                w = - penalty_factor
            reward = w*(max_dist - dist)/ max_dist
            self.curr_reward +=reward
        self.last_goal_dist = dist

        

    def _reward_collision(
        self,goal_in_robot_frame, dist_to_human: np.ndarray, punishment: float = 10, *args, **kwargs
    ):
        """
        Reward for colliding with an obstacle.

        :param dist_to_human (np.ndarray): dist_to_human data
        :param punishment (float, optional): punishment for collision. defaults to 10
        """
        dist = []
        for i in dist_to_human:
            if i != 0.0:
                dist.append(i)
            else :
                pass

        if len(dist) != 0:
            if np.array(dist).min() <= goal_in_robot_frame[2]:
                self.curr_reward -= punishment
                self.info["crash"] = True
                self.info["is_done"] = True
                self.info["is_success"] = 0
                self.info["done_reason"] = 1
            else: 
                pass
        else:
            pass

    def _reward_safe_dist(
        self, dist_to_human: np.ndarray, punishment: float = 0.15, *args, **kwargs
    ):
        """
        Reward for undercutting safe distance.

        :param dist_to_human (np.ndarray): dist_to_human data
        :param punishment (float, optional): punishment for undercutting. defaults to 0.15
        """
        dist = []
        for i in dist_to_human:
            if i != 0.0:
                dist.append(i)
            else :
                pass
        if len(dist) != 0:
            if np.array(dist).min() < self.safe_dist:
                self.curr_reward -= punishment
            else: 
                pass
        else:
            pass
