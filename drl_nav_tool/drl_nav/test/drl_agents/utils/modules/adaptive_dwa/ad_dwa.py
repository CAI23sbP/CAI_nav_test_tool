import math

import matplotlib.pyplot as plt
import numpy as np

class DWA():
    def __init__(self, config):
        self.config = config

    def motion(self, x, u, dt):
        """
        motion model
        """
        # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x
    
    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.config.min_speed, self.config.max_speed,
            -self.config.max_yaw_rate, self.config.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.config.max_accel * self.config.dt,
            x[3] + self.config.max_accel * self.config.dt,
            x[4] - self.config.max_delta_yaw_rate * self.config.dt,
            x[4] + self.config.max_delta_yaw_rate * self.config.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw
    
    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.config.predict_time:
            x = self.motion(x, [v, y], self.config.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.config.dt

        return trajectory

    def calc_control_and_trajectory(self, x, dw, goal, ob, param, radius):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.config.v_resolution):
            for y in np.arange(dw[2], dw[3], self.config.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y)
                # calc cost
                to_goal_cost = param[0] * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = param[1] * (self.config.max_speed - trajectory[-1, 3])
                ob_cost = param[2] * self.calc_obstacle_cost(trajectory, ob, radius)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.config.robot_stuck_flag_cons \
                            and abs(x[3]) < self.config.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.config.max_delta_yaw_rate
        return best_u, best_trajectory
    

    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost
    

    def calc_obstacle_cost(self, trajectory, ob, radius):
        """
        calc obstacle cost inf: collision
        """
        

        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        if self.config.robot_type == "rectangle":
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= self.config.robot_length / 2
            right_check = local_ob[:, 1] <= self.config.robot_width / 2
            bottom_check = local_ob[:, 0] >= -self.config.robot_length / 2
            left_check = local_ob[:, 1] >= -self.config.robot_width / 2
            if (np.logical_and(np.logical_and(upper_check, right_check),
                            np.logical_and(bottom_check, left_check))).any():
                return float("Inf")
            
        elif self.config.robot_type == "circle":
            if np.array(r <= radius).any():
                return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK
    
    def dwa_control(self, x, goal, ob, param, radius):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, ob, param, radius)

        return u, trajectory

    def move(self, state, param):
        radius = state["robot_state"][2]
        vel = state["robot_state"][3]
        angle = state["robot_state"][4]
        gx = state["robot_state"][5]
        gy = state["robot_state"][6]
        c_x = state["robot_state"][7]
        c_y = state["robot_state"][8]
        yaw = state["robot_state"][9]

        ob = np.array(state["scan_pose"])

        # init # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        x = np.array([c_x, c_y, yaw, vel, angle])

        goal = np.array([gx, gy])
        u, _ = self.dwa_control(x, goal, ob, param, radius)
        x = self.motion(x, u, self.config.dt) 

        return x[3],x[4]