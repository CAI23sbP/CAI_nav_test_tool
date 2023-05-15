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

from std_msgs.msg import ColorRGBA, Int16
from visualization_msgs.msg import MarkerArray,Marker
from tf.transformations import  euler_from_quaternion
import os,sys
sys.path.append(os.environ["SIM_PKG"]+'/drl_nav_tool/drl_nav/utils/')
from config import Constants

class ObservationManager:

    """
    In this part, a observation to use in DRL or else
    """
    def __init__(self):
        # tmgr
        # last updated topic
        self.update_cluster = True
        self.range = None
        self.header = Header()
        self.robot_cmd_x = 0
        self.robot_cmd_y = 0
        self.robot_pos_x ,self.robot_pos_y= 0,0
        self.robot_yaw=0
        self.n_reset = -1
        self.obstacles = {}
        self.cluster = Clusters()
        self.markers = MarkerArray()
        # sub
        self.sub_vel =  rospy.Subscriber("/burger/cmd_vel", Twist, self.cmd_vel)
        self.sub_odom = rospy.Subscriber("/burger/odom",Odometry,self.odom)
        self.sub_reset = rospy.Subscriber("/scenario_reset", Int16, self.cb_reset)
        self.sub_header = rospy.Subscriber("/visualization_marker", Marker,self.sub_header)



        self.pub_obst_odom = rospy.Publisher("/obst_odom", Clusters, queue_size=1)
        self.pub_timer = rospy.Timer(rospy.Duration(0.1), self.pub_odom)
        if Constants.RobotManager.IS_GT == False:
            self.sub_scan = rospy.Subscriber("/burger/scan",LaserScan, self.scan_range)
            self.scan_range_vis = rospy.Publisher("/vis_observaion_range",MarkerArray,queue_size=1)
            self.pub_vis_human = rospy.Publisher("/visible_human",MarkerArray,queue_size=1)
        # pub
        
        self.cb_reset(0)


    def sub_header(self,msg):
        self.header=msg.header

    def cmd_vel(self,msg):
        self.robot_cmd_x = msg.linear.x
        self.robot_cmd_y = msg.linear.y

    def cb_reset(self, msg):
        # collect static and dynamic obstacles
        self.n_reset += 1
        self.obst_topics = []
        self.get_obstacle_topics()

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
        marker.pose.position.x = self.robot_pos_x 
        marker.pose.position.y = self.robot_pos_y 
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
        marker.pose.position.x = self.robot_pos_x
        marker.pose.position.y = self.robot_pos_y

        return marker

    def update_obstacle_odom(self):
        # subscriber
        # debug objects have unique topics
        for topic in self.obst_topics:
            rospy.Subscriber(topic, MarkerArray, self.cb_marker, topic)
        # pedsim agents are all collected in one topic
        rospy.Subscriber(
            "/pedsim_simulator/simulated_agents", peds.AgentStates, self.cb_marker
        )
    
    def odom(self,msg):
        self.robot_pos_x = msg.pose.pose.position.x
        self.robot_pos_y = msg.pose.pose.position.y
        _,_ , self.robot_yaw = euler_from_quaternion([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])


    def get_obstacle_topics(self):
        topics = rospy.get_published_topics()
        for t_list in topics:
            for t in t_list:
                if "/debug/model/obs_" in t:
                    self.obst_topics.append(t)
        self.update_obstacle_odom()

    def pub_odom(self, event):
        self.update_cluster = False
        self.fill_cluster()
        

        self.pub_obst_odom.publish(self.cluster)
        # reset cluster
        self.markers.markers.clear()
        self.update_cluster = True

    def scan_range(self,msg):
        self.range=msg.range_max
        
    def remove_covered(self):
        pass
    
    def fill_cluster_dist(self):
        self.markers.markers.append(self.vis_robot())
        self.cluster = Clusters()
        for i, topic in enumerate(self.obstacles):
            x = self.obstacles[topic][0].x - self.robot_pos_x 
            y = self.obstacles[topic][0].y - self.robot_pos_y
            distance= np.linalg.norm([x , y],2)
            if distance < self.range:
                tmp_point = Point()
                tmp_vel = Vector3()

                tmp_point.x = x 
                tmp_point.y = y
                tmp_point.z = self.obstacles[topic][1] - self.robot_yaw
                tmp_vel.x = self.obstacles[topic][2].x - self.robot_cmd_x
                
                tmp_vel.y = self.obstacles[topic][2].y - self.robot_cmd_y
                self.markers.markers.append(self.vis_human(topic,is_vis=True))
                self.cluster.mean_points.append(tmp_point)
                self.cluster.velocities.append(tmp_vel)
                self.cluster.labels.append(i)
            else:
                self.markers.markers.append(self.vis_human(topic,is_vis=False))
            self.pub_vis_human.publish(self.markers)
            
    def fill_cluster_gt(self):
        
        self.markers.markers.append(self.vis_robot())
        self.cluster = Clusters()
        for i, topic in enumerate(self.obstacles):
    
            tmp_point = Point()
            tmp_vel = Vector3() 

            tmp_point.x = self.obstacles[topic][0].x - self.robot_pos_x 
            tmp_point.y = self.obstacles[topic][0].y - self.robot_pos_y
            tmp_point.z = self.obstacles[topic][1] - self.robot_yaw
            tmp_vel.x = self.obstacles[topic][2].x-self.robot_cmd_x
            tmp_vel.y = self.obstacles[topic][2].y-self.robot_cmd_y

            self.cluster.mean_points.append(tmp_point)
            self.cluster.velocities.append(tmp_vel)
            self.cluster.labels.append(i)
            self.markers.markers.append(self.vis_human(topic,is_vis=True))

        self.pub_vis_human.publish(self.markers)


    def fill_cluster(self):
        
        if self.range!=None:
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


def run():
    rospy.init_node("observation", anonymous=False)
    ObservationManager()
    rospy.spin()


if __name__ == "__main__":
    run()
