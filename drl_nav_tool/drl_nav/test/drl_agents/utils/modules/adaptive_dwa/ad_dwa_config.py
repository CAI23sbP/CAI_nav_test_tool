import math

class DWA_Config:
    """
    simulation parameter class
    """

        # robot parameter
    max_speed = 1.2  # [m/s]
    min_speed = -0.5  # [m/s]
    max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
    max_accel = 0.2  # [m/ss]
    max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
    v_resolution = 0.01  # [m/s]
    yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
    dt = 0.1  # [s] Time tick for motion prediction #TO set same about simulator rate
    predict_time = 1.0  # [s]

    robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
# Also used to check if goal is reached in both types
    robot_radius = 1.0  # [m] for collision check
    robot_type = "circle"

       