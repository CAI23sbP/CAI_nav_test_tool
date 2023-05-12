# CAI_nav_test_tool
### "We have modified [arena_rosnav](https://github.com/Arena-Rosnav/arena-rosnav) to make it easy for users to customize." ###

# 1. INFORMATION #


IF you want to train your agent,you should go to under link 
[https://github.com/CAI23sbP/CAI_nav_train_tool]


# 2. How to install #

`mkdir test_ws && cd test_ws && mkdir src`

`cd src && git clone https://github.com/CAI23sbP/CAI_nav_test_tool.git`

`cd .. && rosdep install --from-paths src --ignore-src --rosdistro noetic -y`

`catkin_make`

`sudo gedit ~/.bashrc`

`export SIM_PKG="<$your home>/test_ws/src/CAI_nav_test_tool"`

`source ~/test_ws/devel/setup.bash`

`source ~/.bashrc`

`source devel/setup.bash`

`cd src && cd CAI_nav_test_tool && pip install -r requirements.txt`



## If you want to test ds-rnn, you should follow under line. ##

`pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102`

`git clone https://github.com/openai/baselines.git`

`cd baselines`

`pip install -e .`



# 3. How to launching #
### To testing ###
`roslaunch simulation_bringup test_model.launch`

![Screenshot from 2023-04-15 22-56-17](https://user-images.githubusercontent.com/108871750/232229143-cc9af4c2-f793-4f13-a7db-dd9586743ce6.png)![Screenshot from 2023-04-15 22-59-39](https://user-images.githubusercontent.com/108871750/232229144-dae92f6c-d2e8-4831-98ae-24c575497aed.png)



### To evaluation ###
Go to evaluation folder.

`cd /test_ws/src/nav_test_tool/evaluation`

Get metrix for evaluation.

`python3 python3 get_metrics.py --name {data folder name [ex) 09-05-2023_20-56-06_burger] }`

You must modify a "sample_schema.yaml" file which is in plot_declarations like under line.

`datasets: [09-05-2023_20-56-06_burger]`

Get evaluation results.

`python3 create_plots.py sample_schema.yml`



# 4. How to modifying config #
You should go to `cd sim_ws/src/nav_sim_test/drl_nav_tool/utils` ,and open <config.py> by "a source code editor"


![path.pdf](https://github.com/CAI23sbP/nav_test_tool/files/11452326/path.pdf)

ETC....


# 5. TODO #
- [X] Spawn static obstacles in the flatland and the pedsim
- [x] Make a perfect evaluation code  


- [ ] Put rosparams or params into config.py  (?)
- [ ] Spawn ORCA in the flatland  (?)


# CONTACT US #
[http://cai.hanyang.ac.kr/] 
[sbp0783@naver.com]
