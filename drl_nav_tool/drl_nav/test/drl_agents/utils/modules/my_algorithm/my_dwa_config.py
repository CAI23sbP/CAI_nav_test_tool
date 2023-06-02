import math

class DWA_Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction #TO set same about simulator rate
        self.predict_time = 3.0  # [s]

        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check
        self.robot_type = "circle"

       