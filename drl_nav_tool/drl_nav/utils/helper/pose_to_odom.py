
# import rospy

# from nav_msgs.msg import Odometry


# class PoseToOdom:
#     def __init__(self):
#         rospy.Subscriber("/burger_0_0/odom", Odometry, self.odom_sub)
#         self.odom_pub = rospy.Publisher("/pedsim_simulator/robot_position", Odometry, queue_size=10)

#         self.seq = 0

#     def odom_sub(self, data):
        
#         self.odom_pub.publish(data)

# if __name__ == "__main__":
#     rospy.init_node("pose_to_odom", anonymous=False)

#     pose_to_odom = PoseToOdom()

#     rospy.spin()
