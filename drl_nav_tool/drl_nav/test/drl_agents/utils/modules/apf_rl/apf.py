import numpy as np
import matplotlib.pyplot as plt
import rospy 
import math

class APF():
    def __init__(self, config):
        self.kp_att = config.kp_att
        self.kp_rel = config.kp_rel
        self.obstacle_bound = config.obstacle_bound  
        self.dt = config.dt
        self.init = None
    

    
    def motion(self, x, dt):
        """
        motion model
        """
        dist = np.linalg.norm([x[0], x[1]])
        return np.array([-dist *math.sin(x[2]), dist *math.cos(x[2])]) #-> linear vel, angular vel
    
    @staticmethod
    def derivative (beta_x,beta_y, beta,  d, d_x, d_y, b, param, h):
        f = (d/((d**param+beta*b))**(1/param))
        f_x = d_x/((d_x**param+beta_x*b))**(1/param)
        f_y = d_y/((d_y**param+beta_y*b))**(1/param)
        return np.array([-(f_x - f) / h ,-(f_y - f) / h])
   
    @staticmethod
    def calc_attractive_force( x, y, gx, gy, range_ ,h):
        e_x, e_y = gx-x, gy-y
        e_x_x, e_y_y = gx-x-h, gy-y-h

        beta_ =  range_**2
        
        distance = np.linalg.norm([e_x,e_y])**2
        distance_x = np.linalg.norm([e_x_x,e_y])**2
        distance_y = np.linalg.norm([e_x,e_y_y])**2

        return distance**2 ,distance_x**2 ,distance_y**2 ,beta_ 
  
    @staticmethod
    def calc_repulsive_force( radius, obs , h):
        multiply_x ,multiply_y , multiply= 1, 1, 1

        for i in range(len(obs)):
                multiply_x = multiply_x*(np.linalg.norm([obs[:,0][i]-h,obs[:,1][i]]) - radius**2) 
                multiply_y = multiply_x*(np.linalg.norm([obs[:,0][i],obs[:,1][i]]-h) - radius**2) 
                multiply = multiply*(np.linalg.norm([obs[:,0][i],obs[:,1][i]]) - radius**2) 
        return multiply_x, multiply_y , multiply
    
    def Navigation_function(self,x, y, gx, gy, obs, radius, param, range_):
        h=1e-6
        APF.calc_attractive_force(x+h, y, gx, gy, range_)
        d, d_x, d_y, b = APF.calc_attractive_force(x+h, y, gx, gy, range_, h=1e-6)
        beta_x,beta_y, beta = APF.calc_repulsive_force(radius,  obs, h=1e-6)

        return APF.derivative(beta_x,beta_y, beta,  d, d_x, d_y, b, param, h=1e-6)

    def move(self, state, param):
        ###TODO
        #    0    1      2           3             4             5             6             7             8              9        10     11        12
        # [rho, theta, inter_rho, inter_theta ,robot_pose.x, robot_pose.y, globalplan.x, globalplan.y, interplan.x, interplan.y ,0.3,  scan_range, theta ]
        
        # yaw = state["robot_state"][5]
        param = np.clip(param, a_min = 0.00001 , a_max = np.inf) ## K parameter must be bigger than zeros

        c_x = state["robot_state"][4]
        c_y = state["robot_state"][5]
        intgx = state["robot_state"][8]
        intgy = state["robot_state"][9]
        radius = state["robot_state"][10]
        range_ = state["robot_state"][11]
        theta = state["robot_state"][12]
        obs = state["scan_state"] ## obs_x,obs_y
        x = [c_x,c_y, theta]
        velocity = self.Navigation_function(c_x, c_y, intgx, intgy, obs, radius, param, range_) #vx,vy
        linear, anguler = self.motion(x, self.dt) @ velocity

        return linear, anguler 
       

   
    