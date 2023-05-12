#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from ford_msgs.msg import Clusters
import torch
import numpy as np
import os
import gym
import sys
sys.path.append(os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav/test/")
from Base_Agent import BaseAgent
from predictor_ import Predict_Trajectory
sys.path.append(os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav/utils/")
from config import Constants,Test
sys.path.append(os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav/")  
from train.Architectures.Architecture_Tree import ArchitectureTree
from train.Architectures.ppo_model import PPO
from train.Architectures.configs.arguments import get_args
from train.Architectures.configs.config import Config
from tf.transformations import  euler_from_quaternion
import copy 

class TestAgent(BaseAgent):
    """
    This is Testing about trained agent, 
    You must use a network which is a actor-critic based.
    TODO is existing 
    """
    
    def __init__(self):
        self.config = Config()
        self.algo_args = get_args()
        self.set_ob_act_space()
        self.current_human_states = np.ones((self.config.sim.human_num, 2)) * 15
        self.n_reset = -1

        self.last_left = 0.
        self.last_right = 0.
        self.last_w = 0.0

        self.base="selfAttn_merge_srnn"
        self.distance = 0       
        self.desired_action = [0.0,0.0]
        self.global_goal = PoseStamped()
        # setting config
        self.predictor = Predict_Trajectory(config = self.config)

        self.device=Test.DEVICE
      
        self.Model=ArchitectureTree.instantiate(Test.ARCHITECTURE_NAME.upper())
        self.weight_path=Test.WEIGHT_PATH
        self.goal_distance=Test.GOAL_DISTANCE

        # subs
        self.sub_global_goal = rospy.Subscriber(
            'burger/move_base_simple/goal', PoseStamped, self.cbGlobalGoal)
        self.sub_pose = rospy.Subscriber('/burger/odom', Odometry, self.cbPose)
        self.robot_max_vel = Constants.RobotManager.MAX_VEL
        self.robot_radius = Constants.RobotManager.RADIUS
        self.vel_x, self.vel_y = 0,0 ## for init

        self.sub_observation = rospy.Subscriber('/obst_odom',Clusters,self.cbObserv)

        # pubs
        self.pub_twist = rospy.Publisher('burger/cmd_vel', Twist, queue_size=1)

        rospy.sleep(5)
        self.set_model()

        self.nn_timer = rospy.Timer(
            rospy.Duration(0.01), self.cbComputeActionArena)
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)        

    def set_ob_act_space(self):
		# set observation space and action space
		# we set the max and min of action/observation space as inf
		# clip the action and observation as you need

        d = {}
		# robot node: num_visible_humans, px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
		# only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        self.spatial_edge_dim = int(2 * (self.config.sim.predict_steps + 1))
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.human_num + self.config.sim.human_num_range,
                                                    self.spatial_edge_dim),
                                            dtype=np.float32)
   
        # number of humans detected at each timestep

        # masks for gst pred model
        # whether each human is visible to robot
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.human_num + self.config.sim.human_num_range,),
                                            dtype=np.bool_)

        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    def cbGlobalGoal(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

    def cbPose(self,msg):
        self.cur_pos_x = msg.pose.pose.position.x
        self.cur_pos_y = msg.pose.pose.position.y
        _,_ , self.robot_yaw = euler_from_quaternion([msg.pose.pose.orientation.x,
                                                    msg.pose.pose.orientation.y,
                                                    msg.pose.pose.orientation.z,
                                                    msg.pose.pose.orientation.w])

    def cbObserv(self,msg):
        self.human_num = len(msg.mean_points)
        self.human_state = msg

    def goalReached(self):
        
        if self.distance > self.goal_distance: 
            return False
        else:
            return True
        
    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)


    def update_action(self, action):
        self.desired_action = action

    def cbControl(self, event):
        self.performAction(self.desired_action[0],self.desired_action[1])
        return
    
    def clip_action(self, raw_action, v_pref):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        holonomic = True if self.config.action_space.kinematics == 'holonomic' else False
        if holonomic:
            act_norm = np.linalg.norm(raw_action)
            if act_norm > v_pref:
                raw_action[0] = raw_action[0] / act_norm * v_pref
                raw_action[1] = raw_action[1] / act_norm * v_pref
            return raw_action[0], raw_action[1]
        else:
            raw_action[0] = np.clip(raw_action[0], -0.1, 0.087) # action[0] is change of v

            raw_action[1] = np.clip(raw_action[1], -0.06, 0.06) # action[1] is change of theta

            return raw_action[0], raw_action[1]

    def smooth(self, v, w):
        beta = 0.1 #TODO: you use 0.2 in the simulator
        left = (2. * v - 0.23 * w) / (2. * 0.035)
        right = (2. * v + 0.23 * w) / (2. * 0.035)
        left = np.clip(left, -17.5, 17.5)
        right = np.clip(right, -17.5, 17.5)
        left = (1.-beta) * self.last_left + beta * left
        right = (1.-beta) * self.last_right + beta * right
        
        self.last_left = copy.deepcopy(left)
        self.last_right = copy.deepcopy(right)

        v_smooth = 0.035 / 2 * (left + right)
        w_smooth = 0.035 / 0.23 * (right - left)

        return v_smooth, w_smooth

    def performAction(self, action_1,action_2):
        twist = Twist()
        twist.linear.x =action_1
        twist.angular.z = action_2

        self.pub_twist.publish(twist)

    def generate_ob(self,is_reset):
        ob={}
        ob['robot_node'] = torch.tensor([[[self.cur_pos_x,self.cur_pos_y,
                                self.robot_radius,
                                -self.goal_x,-self.goal_y,
                                self.robot_max_vel,
                                self.robot_yaw]]])
        
    
        ob['temporal_edges'] = torch.tensor([[[self.vel_x,self.vel_y]]]) ## robot vel x,y

        ob['spatial_edges'] = torch.ones((self.config.sim.human_num, int(2*(self.config.sim.predict_steps+1)))) * np.inf
        ob['visible_masks'] = torch.zeros((self.config.sim.human_num,), dtype=bool).view(self.config.sim.human_num,1).T
        
        
        hum_state = self.human_state
        for i in range(len(hum_state.mean_points)):
            relative_pos =  torch.tensor([hum_state.mean_points[i].x , hum_state.mean_points[i].y]) ## already relative
            ob['spatial_edges'][hum_state.labels[i], :2] = relative_pos
            ob['visible_masks'][0][hum_state.labels[i]] = True
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        spatial_edges=np.concatenate([self.current_human_states, np.zeros((self.config.sim.human_num, 2 * self.config.sim.predict_steps))], axis=1)
        ob['spatial_edges'] = torch.tensor([spatial_edges.tolist()])
        
        if self.human_num==0:
            ob['detected_human_num'] = torch.tensor([[1]],dtype=float )

        ob['detected_human_num'] = torch.tensor([[self.human_num]],dtype=float )
       
        if is_reset :
            predict_ob = self.predictor.reset(ob)
        else:
            predict_ob = self.predictor.process_obs_rew(ob)

        return predict_ob
    
    def set_model(self):
        ob=self.generate_ob(is_reset=True)
        self.net = self.Model(ob,
                              self.action_space,
                              base_kwargs = self.algo_args,
                              base = self.base)
        self.net.load_state_dict(torch.load(self.weight_path,map_location=torch.device(self.device)))

    def cbComputeActionArena(self, event):
        self.distance=np.linalg.norm(np.array([self.cur_pos_x-self.goal_x,
                                               self.cur_pos_y-self.goal_y]),2) 
        if not self.goalReached():
            
            with torch.no_grad():
                ob = self.generate_ob(is_reset=False)
                eval_recurrent_hidden_states = {}
                node_num = 1
                edge_num = self.net.base.human_num + 1
                eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(1, node_num, self.net.base.human_node_rnn_size * 1)
                eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(1, edge_num,
                                                                            self.net.base.human_human_edge_rnn_size*1)
                eval_masks = torch.zeros(1, 1)
                _, action, _, eval_recurrent_hidden_states = self.net.act(
                        ob[0],
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            action=action[0]
            self.desired_action[0] = np.clip(self.desired_action[0] + action[0], -self.robot_max_vel, self.robot_max_vel)
            self.desired_action[1] = action[1] # TODO: dynamic time step is not supported now
            v_smooth, w_smooth = self.smooth(self.desired_action[0], self.desired_action[1])
            # rospy.loginfo(f"linear: {v_smooth}, angular: {w_smooth}")
            self.performAction(v_smooth, w_smooth)

        else:
            self.stop_moving()
            return
        
    def on_shutdown(self):
        rospy.loginfo("[%s] Shutting down Node.")
        self.stop_moving()

if __name__=="__main__":

    rospy.init_node('test_node', anonymous=False)
    print('==================================\ntest node started\n==================================')

    agent = TestAgent()
    rospy.on_shutdown(agent.on_shutdown)

    rospy.spin()