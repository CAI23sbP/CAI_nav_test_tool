cmake_minimum_required(VERSION 3.0.2)
project(robohumstamap)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  rospy
  std_msgs
  message_generation
  
)


generate_messages(
  DEPENDENCIES
  nav_msgs
  std_msgs
  
)

catkin_package(
  CATKIN_DEPENDS std_msgs message_runtime nav_msgs 
)

catkin_install_python(PROGRAMS

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

