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
import torch
from std_msgs.msg import ColorRGBA, Int16
from visualization_msgs.msg import MarkerArray,Marker
from tf.transformations import  euler_from_quaternion
from geometry_msgs.msg import Pose2D
from ..config import Config 
from .config import DSRNNConfigs

class DsRNNObservationCollector:

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
        self.sub_scan = rospy.Subscriber("/scan",LaserScan, self.scan_range)
        self.scan_range_vis = rospy.Publisher("/vis_observaion_range",MarkerArray,queue_size=1)

        self.pub_obst_odom = rospy.Publisher("/obst_odom", Clusters, queue_size=1)
        self.pub_timer = rospy.Timer(rospy.Duration(0.1), self.pub_odom)
        self.pub_vis_human = rospy.Publisher("/visible_human",MarkerArray,queue_size=1)
        # pub
        
        self._globalplan_sub = rospy.Subscriber(
            "move_base/goal", PoseStamped, self.callback_global_plan
        )
        self.device = DSRNNConfigs.device
        #predict

        self.cb_reset(0)


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
        self.update_cluster = True

    def scan_range(self,msg):
        self.range = msg.range_max
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
        all_spatial_edges = torch.ones((self.max_human_num, 2))*15.0
        detected_human_num = len(cluster.labels)

        if detected_human_num> self.max_human_num:
            print("#############")
            print(f"[detected_human_num]:{detected_human_num}")
            print(f"[cluster.mean_points]:{len(cluster.mean_points)}")
            print(cluster.labels , f"\t {len(cluster.labels)}")
            print("#############")
        for i in range(detected_human_num):
            all_spatial_edges[i,:2] = torch.tensor([cluster.mean_points [i].x,cluster.mean_points [i].y])


        spatial_edges = all_spatial_edges
        spatial_edges[torch.isinf(spatial_edges)] = 15.


        robot_node = [self._robot_pose.x, self._robot_pose.y ,0.3,self._globalplan.x, self._globalplan.y,self._robot_pose.theta ]
        observation = {
                    "robot_node": torch.tensor(robot_node,device=self.device).reshape(1,1,6),
                    "temporal_edges": torch.tensor([self.robot_v.linear.x,self.robot_v.angular.z],device=self.device).reshape(1,1,2),
                    "spatial_edges" :  spatial_edges.reshape(1,self.max_human_num,spatial_edges.shape[1]).to(self.device)

        }
        return observation

    def callback_global_plan(self, msg_global_plan):
        self._globalplan = DsRNNObservationCollector.process_global_plan_msg(
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
