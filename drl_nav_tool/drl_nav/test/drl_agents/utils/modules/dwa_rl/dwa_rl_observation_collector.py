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
from squaternion import Quaternion

from std_msgs.msg import ColorRGBA, Int16
from visualization_msgs.msg import MarkerArray,Marker
from tf.transformations import  euler_from_quaternion
from geometry_msgs.msg import Pose2D
from .dwa_rl_config import DWARLConfig 
import math, time
from .dwa import DWA

class DWA_RL_ObservationCollector:
    def __init__(self):
        self.received_scan = False
        self.received_odom = False
        self.dec_obs = rospy.get_param("/number_decimals_precision_obs", 1)
        self.n_laser_discretization = rospy.get_param('/n_laser_discretization',128)
        self.n_observations = rospy.get_param('/n_observations',144)
        self.min_range = rospy.get_param('/min_range',0.3)
        self.max_cost = rospy.get_param('/max_cost',1)
        self.min_cost = rospy.get_param('/min_cost',0)
        self.n_stacked_frames = rospy.get_param('/n_stacked_frames',10)
        self.n_skipped_frames = rospy.get_param('/n_skipped_frames',4)
        self.max_linear_speed = rospy.get_param('/max_linear_speed',0.65)
        self.max_angular_speed = rospy.get_param('/max_angular_speed',1)
        self.min_linear_speed = rospy.get_param('/min_linear_speed',0)
        self.min_angular_speed = rospy.get_param('/min_angular_speed',0)
        self.n_skipped_count = 0

        self.goal_sub = rospy.Subscriber(
                "move_base/goal", PoseStamped, self.callback_goal
            )
        self.scan_sub = rospy.Subscriber(
                    "/scan",LaserScan, self.callback_scan
            )
        self.odom_sub = rospy.Subscriber(
                    "/odom",Odometry,self.callback_odom
            )
        self.laser_filtered_pub = rospy.Publisher('/laser/scan_filtered', LaserScan, queue_size=1)

    def wait_for_scan_and_odom(self):
        while not self.received_odom or not self.received_scan :
            pass

        self.received_odom = False
        self.received_scan = False

    def callback_goal(self,msg):
        self.goal = msg.pose.position
         
    def callback_scan(self,msg):
        self.received_scan = True
        self.scan = msg
        self.new_ranges = int(math.ceil(float(len(self.scan.ranges)) / float(self.n_laser_discretization)))
    
    def callback_odom(self,msg):
        self.received_odom = True
        self.odom = msg

    def discretize_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        discretized_ranges = []
        filtered_range = []
        mod = new_ranges
        max_laser_value = data.range_max
        min_laser_value = data.range_min

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    #discretized_ranges.append(self.max_laser_value)
                    discretized_ranges.append(round(max_laser_value,self.dec_obs))
                elif np.isnan(item):
                    #discretized_ranges.append(self.min_laser_value)
                    discretized_ranges.append(round(min_laser_value,self.dec_obs))
                else:
                    #discretized_ranges.append(int(item))
                    discretized_ranges.append(round(item,self.dec_obs))

                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.1)
       
        return discretized_ranges
    
    def _get_distance2goal(self):
        """ Gets the distance to the goal
        """
        return math.sqrt((self.goal.x - self.odom.pose.pose.position.x)**2 + (self.goal.y - self.odom.pose.pose.position.y)**2)

    def get_observations(self, *args, **kwargs):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        if kwargs.get("wait_for_messages"):
            self.wait_for_scan_and_odom()

        discretized_observations = self.discretize_observation( self.scan,
                                                                self.new_ranges
                                                                )


        discretized_observations = np.asarray(discretized_observations)

        # New code for getting observations based on DWA
        odom_data = self.odom

        q = Quaternion(odom_data.pose.pose.orientation.w,odom_data.pose.pose.orientation.x,odom_data.pose.pose.orientation.y,odom_data.pose.pose.orientation.z)
        e = q.to_euler(degrees=False)
        laser_scan = np.array(self.scan.ranges)
        laser_scan[np.isinf(laser_scan)] = 15.
        odom_dict = {
                "x": odom_data.pose.pose.position.x,
                "y": odom_data.pose.pose.position.y,
                "theta": e[2],
                "u": odom_data.twist.twist.linear.x,
                "omega": odom_data.twist.twist.angular.z
        }
        odom_list = [odom_dict[key] for key in odom_dict]
        
        goal_pose = {
            "x": self.goal.x,
            "y": self.goal.y
        }
        cnfg = DWARLConfig(odom_dict, goal_pose)
        self.obs = Obstacles(laser_scan, cnfg) 
        array_generator = (self.obs.obst for _ in range(0, self.n_stacked_frames))
        self.obs_list_stacked = np.column_stack(list(array_generator))
        if(self.n_skipped_count == 0):
            self.obs_list_stacked = np.delete(self.obs_list_stacked, (0,1), 1)
            self.obs_list_stacked = np.append(self.obs_list_stacked, self.obs.obst, 1)
            self.n_skipped_count += 1

        elif(self.n_skipped_count < self.n_skipped_frames):
            self.obs_list_stacked[:, ((2*self.n_stacked_frames)-2):((2*self.n_stacked_frames))] = self.obs.obst
            self.n_skipped_count += 1

        elif(self.n_skipped_count == self.n_skipped_frames):
            self.obs_list_stacked[:, ((2*self.n_stacked_frames)-2):((2*self.n_stacked_frames))] = self.obs.obst
            self.n_skipped_count = 0
        
        self.v_matrix, self.w_matrix, self.cost_matrix, self.obst_cost_matrix, self.to_goal_cost_matrix = DWA(cnfg, self.obs_list_stacked, self.n_stacked_frames)

        self.stacked_obs = np.stack((self.v_matrix, self.w_matrix, self.obst_cost_matrix, self.to_goal_cost_matrix), axis=2)

        distance = self._get_distance2goal()
        return {'stacked_obs': self.stacked_obs,'distance': np.array(distance) ,'scan':np.array(laser_scan),'odom_dict':np.array(odom_list),'v_matrix':self.v_matrix,'w_matrix':self.w_matrix}
    
    
    
import math
import numpy as np
class Obstacles():
    def __init__(self, ranges, config):
        # Set of coordinates of obstacles in view
        self.obst = np.array([])
        self.ranges = ranges
        self.config = config
        self.assignObs()
        
    def myRange(self,start,end,step):
        i = start
        while i < end:
            yield i
            i += step
        yield end

    # Callback for LaserScan
    def assignObs(self):

        deg = len(self.ranges)   # Number of degrees - varies in Sim vs real world
        # print("Laser degree length {}".format(deg))
        self.obst = np.empty([0,2])   # reset the obstacle set to only keep visible objects

        maxAngle = 270
        scanSkip = 4
        anglePerSlot = (float(maxAngle) / deg) * scanSkip
        angleCount = 0
        angleValuePos = 0
        angleValueNeg = 0
        for angle in self.myRange(0,deg-1,scanSkip):
            distance = self.ranges[angle]
            
            if(angleCount < (deg / (2*scanSkip))):
                # print("In negative angle zone")
                angleValueNeg += (anglePerSlot)  
                scanTheta = (angleValueNeg - 135) * math.pi/180.0
                    

            elif(angleCount>(deg / (2*scanSkip))):
                # print("In positive angle zone")
                angleValuePos += anglePerSlot
                scanTheta = angleValuePos * math.pi/180.0
            # only record obstacles that are within 4 metres away

            else:
                scanTheta = 0

            angleCount += 1

            if (distance < 4):

                objTheta =  scanTheta + self.config.th
                obsX = round((self.config.x + (distance * math.cos(abs(objTheta))))*8)/8
                if (objTheta < 0):
                    obsY = round((self.config.y - (distance * math.sin(abs(objTheta))))*8)/8
                else:
                    obsY = round((self.config.y + (distance * math.sin(abs(objTheta))))*8)/8
                obs_row = np.array([obsX, obsY])
                self.obst = np.vstack((self.obst,obs_row))
            else:
                obsX = float("inf")
                obsY = float("inf")
                obs_row = np.array([obsX, obsY])
                self.obst = np.vstack((self.obst,obs_row))



