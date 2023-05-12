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
from .Base_Agent import BaseAgent
sys.path.append(os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav/utils/")
from config import Constants,Test
sys.path.append(os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav/")  
from train.Architectures.Architecture_Tree import ArchitectureTree
from train.Architectures.ppo_model import PPO
from train.Architectures.configs.arguments import get_args
from train.Architectures.configs.config import Config
from tf.transformations import  euler_from_quaternion


class TestAgent(BaseAgent):
    """
    This is Testing about trained agent, 
    You must use a network which is a actor-critic based.
    TODO is existing 
    """
    
    def __init__(self):
        self.config = Config()
        self.algo_args = get_args()
        # self.base_kwargs=self.config
        self.base="srnn"
        self.distance = 0       
        self.desired_action = 0
        self.global_goal = PoseStamped()
        # setting config
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
        self.pub_twist = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        rospy.sleep(5)
        self.nn_timer = rospy.Timer(
            rospy.Duration(0.01), self.cbComputeActionArena)
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)         

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
        self.performAction(self.desired_action)
        return
    
    def performAction(self, action):
        twist = Twist()
        if action==0:
            twist.linear.x=0
            twist.linear.z=0
        else:
            twist.linear.x =action[0]
            twist.angular.z = action[1]

        self.pub_twist.publish(twist)

    def cbComputeActionArena(self, event):
        self.distance=np.linalg.norm(np.array([self.cur_pos_x-self.goal_x,
                                               self.cur_pos_y-self.goal_y]),2) 
        if not self.goalReached():

            ob={}
            ob['robot_node'] = torch.tensor([[[self.cur_pos_x,self.cur_pos_y,
                                self.robot_radius,
                                self.goal_x,self.goal_y,
                                self.robot_max_vel,
                                self.robot_yaw]]])
            print(ob['robot_node'].size())
            ob['temporal_edges'] = torch.tensor([[[self.vel_x,self.vel_y]]]) ## robot vel x,y
            ob['spatial_edges'] = torch.zeros((self.human_num, 2))
            for i in range(self.human_num):
                relative_pos =  torch.tensor([self.human_state.mean_points[i].x , self.human_state.mean_points[i].y]) ## already relative
                ob['spatial_edges'][i] = relative_pos
            ob['spatial_edges']=ob['spatial_edges'].reshape(1, self.human_num, 2)
            
            high = 1.2 * np.ones([2, ])
            self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
            
            net = self.Model(ob,
                              self.action_space,
                              base_kwargs = self.algo_args,
                              base = self.base)

            net.load_state_dict(torch.load(self.weight_path,map_location=torch.device(self.device)))

            eval_recurrent_hidden_states = {}
            node_num = 1
            edge_num = net.base.human_num + 1
            eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(1, node_num, self.config.SRNN.human_node_rnn_size * 1,
                                                                device=self.device)
            eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(1, edge_num,
                                                                       self.config.SRNN.human_human_edge_rnn_size*1,
                                                                device=self.device)
            eval_masks = torch.zeros(1, 1, device=self.device)
            _, action, _, eval_recurrent_hidden_states = net.act(
                    ob,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
            rospy.logerr(action)
            self.performAction(action)

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