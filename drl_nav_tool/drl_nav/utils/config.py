import os 
Global_path= os.environ["SIM_PKG"]+"/drl_nav_tool/drl_nav" 

class Constants:
    """
    you can setting managers and taskes in here 
    """

    STEP_SIZE = 0.1
    # UPDATE_RATE = 10
    # VIZ_PUB_RATE = 30
    IS_TRAIN =  False ##when IS_TRAIN is True, Train is launching

    class RobotManager:
        SPAWN_ROBOT_SAFE_DIST = 0.4
        NAME = "burger"
        MODEL_PATH = Global_path+"/utils/robohumstamap/robot_type/"+NAME+"/"
        MODEL_YAML_PATH = MODEL_PATH+NAME+".model.yaml"
        MAX_VEL = 1.2 #m/s
        RADIUS = 0.4
        LOCAL_PLANNER = "dwa" # [teb, dwa, mpc, rlca, pred_dsrnn, rosnav,etc...]
        AGENT_NAME = "" ## DON'T TOUCH IT
        COMPLEXITY = 1 ## doc="1 = Map known, Position known; 2 = Map known, Position unknown (AMCL); 3 = Map unknown, Position unknown (SLAM)"
        IS_GT = False  ## if True: accept ground true human states

    class ObstacleManager:
        IS_HUMAN = True
        IS_STATIC = False 

        class Human:
            
            HUMAN_TYPE = "PEDHUMAN" 
            HUMAN_MAX_RADIUS = 0.6
            # if HUMAN_TYPE == "PEDHUMAN":
            YAML_FILE = Global_path+"/utils/robohumstamap/human_type/config/ped_config/person_two_legged.model.yaml"
            # elif HUMAN_TYPE =="ORCAHUMAN":
            PARAMS = (10, 10, 5, 5)
            TIME_STEP = 0.25
            V_PREF = 1
            # elif HUMAN_TYPE =="FLATLANDRANDOMHUMAN":
            MIN_RADIUS = 0.2
            MAX_RADIUS = 0.6
            BODY = {
                "name": "base_link",
                "pose": [0, 0, 0],
                "color": [1, 0.2, 0.1, 1.0],
                "footprints": []
            }
            FOOTPRINT = {
                "density": 1,
                "restitution": 1,
                "layers": ["all"],
                "collision": "true",
                "sensor": "false"
            }
            RANDOM_MOVE_PLUGIN = {
            "type": "RandomMove",
            "name": "RandomMove_Plugin",
            "body": "base_link"
            }
            LINEAR_VEL = 1.4
            ANGLUAR_VEL_MAX = 0.7
        
        class Static:
            
            STATIC_TYPE = "PEDSTATIC"    ## must be matched about ped_type and static_type
            # if HUMAN_TYPE == "PEDSTATIC"
            YAML_FILE = Global_path+"/utils/robohumstamap/static_obs_type/config/ped_config/"
            # if HUMAN_TYPE# =="FLATLANDSTATIC":
            MIN_RADIUS = 0.2
            MAX_RADIUS = 3
            BODY = {
                "name": "base_link",
                "pose": [0, 0, 0],
                "color": [1, 0.2, 0.1, 1.0],
                "footprints": []
            }
            FOOTPRINT = {
                "density": 1,
                "restitution": 1,
                "layers": ["all"],
                "collision": "true",
                "sensor": "false"
            }
            RANDOM_MOVE_PLUGIN = {
            "type": "RandomMove",
            "name": "RandomMove_Plugin",
            "body": "base_link"
            }            
            OBSTACLE_MAX_RADIUS = 0.6

    class MapManager:
        MAP_PATH = Global_path+"/utils/robohumstamap/map_type/"
        MAP_TYPE = "default_map"  ## aws_house, bookstore, default_map, evaluation_floor, experiment_room_2, factory, hospital, ignc, map1, office_cpr_construction, small_warehouse, turtlebot3_world 
        MAP_PNG = MAP_TYPE + ".png"

    class TaskMode:
        WAIT_FOR_SERVICE_TIMEOUT = 60
        MAX_RESET_FAIL_TIMES =10
        TIMEOUT = 3.0 * 60

        class Random:
            
            MIN_DYNAMIC_OBS = 5
            MAX_DYNAMIC_OBS = 10
            MIN_STATIC_OBS = 4
            MAX_STATIC_OBS = 5

        class Scenario:
            YAML_PATH = Global_path + "/utils/task_generator/scenario/eval_outdoor_obs20.json"


        """
        you can add class for other class
        """

class Train:
    DEVICE = "cpu"
    ARCHITECTURE_NAME = "PPO"
    WEIGHT_PATH = Global_path+"/train/weight/"+ ARCHITECTURE_NAME+".pt"
    GOAL_DISTANCE = 0.3
    TASK_MODE = "RandomTask" ### Default do not touch it

class Test:
    DEVICE = "cuda"
    ARCHITECTURE_NAME = Train.ARCHITECTURE_NAME
    WEIGHT_PATH = Train.WEIGHT_PATH
    GOAL_DISTANCE = Train.GOAL_DISTANCE
    DESIRED_RESET = 1
    TASK_MODE = "ScenarioTask"  ##ScenarioTask ,RandomTask

    RECORDING = True
    