import numpy as np

class APFRLRewardCalculator:
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
        self.init_pose = []


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

    def get_reward(self,obs_dict):
        self._reset()
        
        scan =obs_dict["scan_dist"]
        robot_node = obs_dict["robot_state"]

        self._cal_reward_rule(state = scan, goal_in_robot_frame = robot_node)
        return self.curr_reward, self.info
    
    @staticmethod
    def transform_robot_frame(robot_node ):
        #[_robot_pose.x, _robot_pose.y, _robot_pose.theta, _globalplan.x, _globalplan.y, _interplan.x, _interplan.y ,0.3,  range]
        cx = robot_node[0].item()
        cy = robot_node[1].item()
        theta= robot_node [2].item()
        px = robot_node[0].item()-robot_node[3].item() ## robot pose_x  - robot goal_x 
        py = robot_node[1].item()-robot_node[4].item()  ## robot pose_y  - robot goal_y
        distance_goal = np.linalg.norm([px,py])   ## distance about current robot pose and goal pose
        radius = robot_node[7].item()  ## robot raidus
        gx = robot_node[3].item() ## goal pose_x
        gy = robot_node[4].item() ## goal pose_y
        return [distance_goal, cx, cy, radius, gx ,gy]
        
    def _cal_reward_rule(self,state ,goal_in_robot_frame):
        """
        reward_goal_reached must be in before reward_collision
        because if you are located reward_collision before reward_goal_reached, then is_done never changed by collision reason 
        """
        self._reward_goal_reached(goal_in_robot_frame, reward=10)
        self._reward_inter_goal_reached(goal_in_robot_frame, reward=10)
        self._reward_goal_approached(goal_in_robot_frame, reward=10)
        self._reward_collision(goal_in_robot_frame,state , punishment=10)
        self._reward_safe_dist(goal_in_robot_frame,state , punishment=0.1)

    def _reward_goal_reached(
        self,
        goal_in_robot_frame,
        reward: float = 10,
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

    def _reward_inter_goal_reached( ##TODO
        self,
        goal_in_robot_frame,
        reward: float = 10,
        *args,
        **kwargs,
    ):
        if goal_in_robot_frame[2] < self.goal_radius:
            self.curr_reward += reward

    def _reward_goal_approached( ##TODO 
        self,
        goal_in_robot_frame,
        reward_factor: float = 2,
        penalty_factor: float = 2,
        *args,
        **kwargs,
    ):
        """
        Reward for approaching the inter_goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward_factor (float, optional): positive factor for approaching goal. defaults to 0.3
        :param penalty_factor (float, optional): negative factor for withdrawing from goal. defaults to 0.5
        """
        if len(self.init_pose) > 1:
            self.init_pose.pop()
        else:     
            self.init_pose.append([goal_in_robot_frame[1], goal_in_robot_frame[2]])
        max_dist = np.linalg.norm([self.init_pose[0][0]-goal_in_robot_frame[4], self.init_pose[0][1]-goal_in_robot_frame[5]])
        dist = goal_in_robot_frame[0]
        if self.last_goal_dist is not None:
            if (self.last_goal_dist - dist) > 0:
                w = reward_factor 
            else:
                w = - penalty_factor
            reward = w*(max_dist - dist)/ max_dist
            self.curr_reward +=reward
        self.last_goal_dist = dist

        

    def _reward_collision(
        self,goal_in_robot_frame, state : np.ndarray, punishment: float = 10, *args, **kwargs
    ):
        """
        Reward for colliding with an obstacle.

        :param dist_to_human (np.ndarray): dist_to_human data
        :param punishment (float, optional): punishment for collision. defaults to 10
        """
      

        if APFRLRewardCalculator.calc_repulsive_force(np.array(state),goal_in_robot_frame) <= goal_in_robot_frame[3]:
            self.curr_reward -= punishment
            self.info["crash"] = True
            self.info["is_done"] = True
            self.info["is_success"] = 0
            self.info["done_reason"] = 1
        else: 
            pass
      

    def _reward_safe_dist(
        self,goal_in_robot_frame, state : np.ndarray, punishment: float = 0.1, *args, **kwargs
    ):
        """
        Reward for undercutting safe distance.

        :param dist_to_human (np.ndarray): dist_to_human data
        :param punishment (float, optional): punishment for undercutting. defaults to 0.15
        """
     
        if APFRLRewardCalculator.calc_repulsive_force(np.array(state),goal_in_robot_frame) < self.safe_dist:
            self.curr_reward -= punishment
        else: 
            pass
    
    @staticmethod
    def calc_repulsive_force(obs, goal_in_robot_frame):

        multiply = 1
      
        for i in range(len(obs)):
                multiply = multiply*(obs[i]**2 - goal_in_robot_frame[3]**2) 
            
        
        return multiply