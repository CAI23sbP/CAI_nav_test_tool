#!/usr/bin/env python3

import pedsim_msgs.msg as peds
import rospy
import numpy as np
from ford_msgs.msg import Clusters
from geometry_msgs.msg import Point, Vector3, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
# from scenario_police import police
from geometry_msgs.msg import PoseStamped
import torch,copy
from std_msgs.msg import ColorRGBA, Int16
from visualization_msgs.msg import MarkerArray,Marker
from tf.transformations import  euler_from_quaternion
from geometry_msgs.msg import Pose2D
from ..config import Config 
from .config import PredictDSRNNConfigs
import os
from architecture.utils.prediction.predictor import CrowdNavPredInter
from architecture.utils.prediction.running_mean_std import RunningMeanStd
from architecture.utils.prediction.temperature_scheduler import Temp_Scheduler
import pickle
from collections import deque

class PredcitDsRNNObservationCollector:

    """
    In this part, a observation to use in DRL or else
    """
    def __init__(self,max_human_num):
        # tmgr
        # last updated topic
        self.update_cluster = True
        self.range = None
        self.header = Header()
        self.robot_cmd_x = 0
        self.robot_cmd_y = 0
        self.n_reset = -1
        self.obstacles = {}
        self.cluster = Clusters()
        self.markers = MarkerArray()
        self.fu_markers = MarkerArray()
        self.received_odom = False
        self.received_scan = False
        self.received_human = False
        self.received_vel = False
        self._globalplan = np.array([])
        self._robot_pose = Pose2D()
        self.max_human_num = max_human_num

        # sub
        self.sub_vel =  rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel)
        self.sub_odom = rospy.Subscriber("/odom",Odometry,self.process_robot_state_msg)
        self.sub_reset = rospy.Subscriber("/scenario_reset", Int16, self.cb_reset)
        self.sub_header = rospy.Subscriber("/visualization_marker", Marker,self.sub_header)

        self.pub_obst_odom = rospy.Publisher("/obst_odom", Clusters, queue_size=1)
        self.pub_timer = rospy.Timer(rospy.Duration(0.1), self.pub_odom)
        self.sub_scan = rospy.Subscriber("/scan",LaserScan, self.scan_range)

        self.scan_range_vis = rospy.Publisher("/vis_observaion_range",MarkerArray,queue_size=1)
        self.pub_vis_human = rospy.Publisher("/visible_human",MarkerArray,queue_size=1)
        self.pub_future_human = rospy.Publisher("/future_human",MarkerArray,queue_size=1)

        # pub
        
        self._globalplan_sub = rospy.Subscriber(
            "move_base/goal", PoseStamped, self.callback_global_plan
        )
        self.device = PredictDSRNNConfigs.device
        #predict
        load_path = os.path.join(os.getcwd(), PredictDSRNNConfigs.load_path)
        checkpoint_dir = os.path.join(load_path, 'checkpoint')
        with open(os.path.join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            self.args = pickle.load(f)
        self.predictor = CrowdNavPredInter(load_path=load_path, device=self.device, config = self.args)
        temperature_scheduler = Temp_Scheduler(self.args.num_epochs, self.args.init_temp, self.args.init_temp, temp_min=0.03)
        self.tau = temperature_scheduler.decay_whole_process(epoch=100)
        self.pred_interval = PredictDSRNNConfigs.pred_interval
        self.buffer_len = (self.args.obs_seq_len - 1) * self.pred_interval + 1

        self.cb_reset(0)

    def predict_ob(self, O, done, rews=0.):
        # O: robot_node: [nenv, 1, 7], spatial_edges: [nenv, observed_human_num, 2*(1+predict_steps)],temporal_edges: [nenv, 1, 2],
        # pos_mask: [nenv, max_human_num], pos_disp_mask: [nenv, max_human_num]
        # prepare inputs for pred_model
        # find humans' absolute positions
        human_pos = O['robot_node'][:, :, :2] + O['spatial_edges'][:, :, :2]
        self.traj_buffer = deque(list(-torch.ones((self.buffer_len, 1, self.max_human_num, 2), device=self.device)*999),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 2)
        self.mask_buffer = deque(list(torch.zeros((self.buffer_len, 1, self.max_human_num, 1), dtype=torch.bool, device=self.device)),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 1)

        # insert the new ob to deque
        self.traj_buffer.append(human_pos)
        self.mask_buffer.append(O['visible_masks'].unsqueeze(-1))
        # [obs_seq_len, nenv, max_human_num, 2] -> [nenv, max_human_num, obs_seq_len, 2]
        in_traj = torch.stack(list(self.traj_buffer)).permute(1, 2, 0, 3)
        in_mask = torch.stack(list(self.mask_buffer)).permute(1, 2, 0, 3).float()

        # index select the input traj and input mask for GST
        in_traj = in_traj[:, :, ::self.pred_interval]
        in_mask = in_mask[:, :, ::self.pred_interval]
        # forward predictor model
        out_traj, out_mask = self.predictor.forward(input_traj=in_traj, input_binary_mask=in_mask)
        out_mask = out_mask.bool()
        # add penalties if the robot collides with predicted future pos of humans
        # deterministic reward, only uses mu_x, mu_y and a predefined radius
        # constant radius of each personal zone circle
        # [nenv, human_num, predict_steps]
        hr_dist_future = out_traj[:, :, :, :2] - O['robot_node'][:, :, :2].unsqueeze(1)
        # [nenv, human_num, predict_steps]
        collision_idx = torch.norm(hr_dist_future, dim=-1) < Config.Env_config.ROBOT_RADIUS + Config.Env_config.Random_model.DYN_OBS_MIN_RADIUS

        # [1,1, predict_steps]
        # mask out invalid predictions
        # [nenv, human_num, predict_steps] AND [nenv, human_num, 1]
        collision_idx = torch.logical_and(collision_idx, out_mask)
        coefficients = 2. ** torch.arange(2, PredictDSRNNConfigs.predict_steps + 2, device=self.device).reshape(
            (1, 1, PredictDSRNNConfigs.predict_steps))  # 4, 8, 16, 32, 64
        

        # [1, 1, predict_steps]
        collision_penalties = -20 / coefficients

        # [nenv, human_num, predict_steps]
        reward_future = collision_idx.to(torch.float)*collision_penalties
        # [nenv, human_num, predict_steps] -> [nenv, human_num*predict_steps] -> [nenv,]
        # keep the values & discard indices
        reward_future, _ = torch.min(reward_future.reshape(1, -1), dim=1)
        # print(reward_future)
        # seems that rews is on cpu
        rews = rews + reward_future.reshape(1, 1).cpu().numpy()

        # get observation back to env
        robot_pos = O['robot_node'][:, :, :2].unsqueeze(1)

        # convert from positions in world frame to robot frame
        out_traj[:, :, :, :2] = out_traj[:, :, :, :2] - robot_pos

        # only take mu_x and mu_y
        out_mask = out_mask.repeat(1, 1, PredictDSRNNConfigs.predict_steps * 2)
        new_spatial_edges = out_traj[:, :, :, :2].reshape(1, self.max_human_num, -1)
 

        O['spatial_edges'][:, :, 2:][out_mask] = new_spatial_edges[out_mask]

        # sort all humans by distance to robot
        # [nenv, human_num]
        hr_dist_cur = torch.linalg.norm(O['spatial_edges'][:, :, :2], dim=-1)
        sorted_idx = torch.argsort(hr_dist_cur, dim=1)

        for i in range(1):
            O['spatial_edges'][i] = O['spatial_edges'][i][sorted_idx[i]]

        obs={'robot_node':O['robot_node'],
            'spatial_edges':O['spatial_edges'],
            'temporal_edges':O['temporal_edges'],
            'visible_masks':O['visible_masks'],
             'detected_human_num': O['detected_human_num'],

        }
        # print(f"[predict]: {obs['spatial_edges']}")

        self.last_pos = copy.deepcopy(human_pos)
        # self.fill_cluster_future(obs['spatial_edges'][0].cpu(),obs['visible_masks'].cpu())
        return obs, rews

    def sub_header(self,msg):
        self.header=msg.header
        
    def cmd_vel(self,msg):
        self.received_vel = True
        self.robot_v = msg
        
    

    def cb_reset(self, msg):
        # collect static and dynamic obstacles
        self.n_reset += 1
        self.obst_topics = []
        self.get_obstacle_topics()


    def update_obstacle_odom(self):
        # subscriber
        # debug objects have unique topics
        for topic in self.obst_topics:
            rospy.Subscriber(topic, MarkerArray, self.cb_marker, topic)
        # pedsim agents are all collected in one topic
        rospy.Subscriber(
            "/pedsim_simulator/simulated_agents", peds.AgentStates, self.cb_marker
        )

    def get_obstacle_topics(self):
        topics = rospy.get_published_topics()
        for t_list in topics:
            for t in t_list:
                if "/flatland_server/debug/model/obs_" in t:
                    self.obst_topics.append(t)
        self.update_obstacle_odom()
        self.received_human = True

    def pub_odom(self, event):
        self.update_cluster = False
        self.fill_cluster()
        
        self.pub_obst_odom.publish(self.cluster)
        # reset cluster
        self.markers.markers.clear()
        self.fu_markers.markers.clear()
        self.update_cluster = True

    def scan_range(self,msg):
        self.range=msg.range_max
        self.received_scan = True
        
    def fill_cluster_dist(self):
        self.markers.markers.append(self.vis_robot())
        self.cluster = Clusters()
        
        for i, topic in enumerate(self.obstacles):
            x = self.obstacles[topic][0].x - self._robot_pose.x 
            y = self.obstacles[topic][0].y - self._robot_pose.y
            distance= np.linalg.norm([x , y],2)
            if distance < self.range:
                tmp_point = Point()
                tmp_vel = Vector3()

                tmp_point.x = x 
                tmp_point.y = y
                tmp_point.z = self.obstacles[topic][1] 

                tmp_vel.x = self.obstacles[topic][2].x 
                tmp_vel.y = self.obstacles[topic][2].y 

                self.cluster.mean_points.append(tmp_point)
                self.cluster.velocities.append(tmp_vel)
                self.cluster.labels.append(i)
                self.markers.markers.append(self.vis_human(topic,is_vis=True))
            else:
                self.markers.markers.append(self.vis_human(topic,is_vis=False))
            self.pub_vis_human.publish(self.markers)
            
        return self.cluster
    
    # def fill_cluster_future(self,out_traj,vis_mask): ##TODO
    #     # self.cluster = Clusters()
    #     for i in np.where(vis_mask.numpy()[0])[0]:
    #         pose = out_traj[i].reshape(int(len(out_traj[i])/2),2) 
    #         for index , j in  enumerate(pose):
    #             tmp_point = Point()
    #             tmp_point.x  = j[0].item()+ self._robot_pose.x
    #             tmp_point.y  = j[1].item()+ self._robot_pose.y 
    #             self.fu_markers.markers.append(self.vis_future_human(i+index,tmp_point))
    #     self.pub_future_human.publish(self.fu_markers)  

    
    # def vis_future_human(self,index,tmp_point):
    #     marker = Marker()
    #     r, g, b, a = [0.5, 0.5, 0.5, 1.0]
    #     marker.ns = f"{index}"
    #     marker.header = self.header
    #     marker.action = Marker.MODIFY
    #     marker.type = Marker.SPHERE
    #     marker.scale = Vector3(1,1,0.1)
    #     marker.color = ColorRGBA(r, g, b, a)
    #     marker.lifetime = rospy.Duration(1)
    #     marker.header.frame_id = "map"
    #     marker.pose.position = tmp_point
    
    #     return marker
    
    def fill_cluster_gt(self):
        self.markers.markers.append(self.vis_robot())
        self.cluster = Clusters()

        for i, topic in enumerate(self.obstacles):
    
            tmp_point = Point()
            tmp_vel = Vector3() 

            tmp_point.x = self.obstacles[topic][0].x - self._robot_pose.x
            tmp_point.y = self.obstacles[topic][0].y - self._robot_pose.y
            tmp_point.z = self.obstacles[topic][1]
            tmp_vel.x = self.obstacles[topic][2].x
            tmp_vel.y = self.obstacles[topic][2].y

            self.cluster.mean_points.append(tmp_point)
            self.cluster.velocities.append(tmp_vel)
            self.cluster.labels.append(i)
            self.markers.markers.append(self.vis_human(topic,is_vis=True))
        self.pub_vis_human.publish(self.markers)

    def wait_for_scan_and_odom(self):
        while not self.received_odom and not self.received_scan and not self.received_human and self.received_vel:
            print("waiting")

            pass

        self.received_odom = False
        self.received_scan = False
        self.received_human = False
        self.received_vel = False

    ##TODO
    # [detected_human_num]:21
    # [cluster.mean_points]:21
    # [0, 1, 4, 5, 8, 10, 11, 12, 13, 16, 17, 18, 19, 10, 11, 12, 13, 16, 17, 18, 19] 	 21
    def get_observations(self, *args, **kwargs):
        if kwargs.get("wait_for_messages"):
            self.wait_for_scan_and_odom()
        observation = {}
        
        cluster = self.fill_cluster_dist()
        all_spatial_edges = torch.ones((self.max_human_num, 2)) * np.inf
        visible_masks = np.zeros(self.max_human_num, dtype=np.bool_)
        detected_human_num = len(cluster.labels)


        # print("#############")
        # print(f"[detected_human_num]:{detected_human_num}")
        # print(f"[cluster.mean_points]:{len(cluster.mean_points)}")
        # print(cluster.labels , f"\t {len(cluster.labels)}")
        # print("#############")
        for i in range(detected_human_num):
            all_spatial_edges[i,:2] = torch.tensor([cluster.mean_points [i].x,cluster.mean_points [i].y])
            visible_masks[i] = True



        spatial_edges = all_spatial_edges
        spatial_edges[torch.isinf(spatial_edges)] = 15


        pre_spatial_edges = np.tile(spatial_edges,PredictDSRNNConfigs.predict_steps+1)
        robot_node = [self._robot_pose.x, self._robot_pose.y ,0.3,self._globalplan.x, self._globalplan.y,self._robot_pose.theta ]

        observation['robot_node'] = torch.tensor(robot_node,device=self.device).reshape(1,1,6)
        observation['temporal_edges'] = torch.tensor([self.robot_v.linear.x,self.robot_v.angular.z],device=self.device).reshape(1,1,2)
        observation['spatial_edges']  = torch.from_numpy(pre_spatial_edges).reshape(1,self.max_human_num,pre_spatial_edges.shape[1]).to(self.device)
                # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if detected_human_num == 0:
            detected_human_num = 1
            observation['detected_human_num'] = torch.tensor([detected_human_num]).reshape(1,1).to(self.device)
        else:
            observation['detected_human_num'] = torch.tensor([detected_human_num]).reshape(1,1).to(self.device)

        observation['visible_masks'] = torch.tensor(visible_masks).reshape(1,self.max_human_num).to(self.device)
            
        obs, rew =self.predict_ob(observation,np.zeros(1))

        return obs,rew

    def callback_global_plan(self, msg_global_plan):
        self._globalplan = PredcitDsRNNObservationCollector.process_global_plan_msg(
            msg_global_plan
        )
        return

    def process_robot_state_msg(self, msg):
        self.received_odom = True
        pose3d = msg.pose.pose
        self._robot_pose = self.pose3D_to_pose2D(pose3d)

    @staticmethod
    def process_global_plan_msg(globalplan):
         
        return globalplan.pose.position
    
    @staticmethod
    def _get_goal_pose_in_robot_frame(goal_pos: Pose2D, robot_pos: Pose2D):
        y_relative = goal_pos.y - robot_pos.y
        x_relative = goal_pos.x- robot_pos.x
        rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
        theta = (
            np.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * np.pi
        ) % (2 * np.pi) - np.pi
        return rho, theta
    
    @staticmethod
    def pose3D_to_pose2D(pose3d):
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (
            pose3d.orientation.x,
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w,
        )
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d
    
    def fill_cluster(self):
        if self.received_scan:

            if Config.Training_config.IS_GT == False: 
                self.vis_ob_range()
                self.fill_cluster_dist()

            else:          
                self.fill_cluster_gt()

    def cb_marker(self, msg, topic=None):

        if self.update_cluster:

            if type(msg) == MarkerArray:
                v = Vector3()
                m = msg.markers[0]
                pos = m.pose.position
                r = m.scale.x / 2
                label = 0
                if topic in self.obstacles:
                    old_pos = self.obstacles[topic][0]
                    old_time = self.obstacles[topic][3].nsecs
                    curr_time = m.header.stamp.nsecs
                    dt = (curr_time - old_time) * 10**-9
                    if dt > 0:
                        v.x = round((pos.x - old_pos.x) / dt, 3)
                        v.y = round((pos.y - old_pos.y) / dt, 3)
                    label = len(self.obst_topics)
                self.obstacles[topic] = [pos, r, v, m.header.stamp, label]
            else:  # Process pedsim agents
                for agent in msg.agent_states:

                    v = agent.twist.linear
                    pos = agent.pose.position
                    _,_, yaw= euler_from_quaternion([agent.pose.orientation.x,agent.pose.orientation.y,agent.pose.orientation.z,agent.pose.orientation.w])
                    label = agent.id
                    self.obstacles[label] = [
                        pos,
                        yaw,
                        v,
                        agent.header.stamp,
                        label + len(self.obst_topics),
                    ]


    def vis_ob_range(self):
        marker = Marker()
        r, g, b, a = [0.9, 0.1, 0.1, 0.1]
        marker.header = self.header
        marker.ns = "scan_range"
        marker.action = Marker.MODIFY
        marker.type = Marker.SPHERE
        marker.scale = Vector3(self.range*2,self.range*2,0.1)
        marker.color = ColorRGBA(r, g, b, a)
        marker.lifetime = rospy.Duration(1)
        marker.header.frame_id = "map"
        marker.pose.position.x = self._robot_pose.x 
        marker.pose.position.y = self._robot_pose.y
        self.scan_range_vis.publish([marker])
      
    def vis_human(self,index,is_vis):
        marker = Marker()
        if is_vis:
            r, g, b, a = [0.1, 0.1, 0.9, 1.0]
        else:
            r, g, b, a = [0.1, 0.9, 0.1, 1.0]

        marker.header = self.header
        marker.ns = f"{index}"
        marker.action = Marker.MODIFY
        marker.type = Marker.SPHERE
        marker.scale = Vector3(1,1,0.1)
        marker.color = ColorRGBA(r, g, b, a)
        marker.lifetime = rospy.Duration(1)
        marker.header.frame_id = "map"
        marker.pose.position = self.obstacles[index][0]

        return marker

    def vis_robot(self):
        marker = Marker()
        r, g, b, a = [0.9, 0.1, 0.1, 1.0]
        marker.header = self.header
        marker.ns = f"robot"
        marker.action = Marker.MODIFY
        marker.type = Marker.SPHERE
        marker.scale = Vector3(1,1,0.1)
        marker.color = ColorRGBA(r, g, b, a)
        marker.lifetime = rospy.Duration(1)
        marker.header.frame_id = "map"
        marker.pose.position.x = self._robot_pose.x
        marker.pose.position.y = self._robot_pose.y

        return marker
