cmake_minimum_required(VERSION 3.5)
project(planner)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(std_srvs REQUIRED)



include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(greedy_planner_node
  src/greedy_planner_node.cpp
  src/hybrid_astar_core.cpp
  src/planner_utils.cpp
)

ament_target_dependencies(greedy_planner_node
  rclcpp
  nav_msgs
  std_msgs
  geometry_msgs
  ament_index_cpp
  example_interfaces
  OpenCV
  std_srvs
)

ament_export_dependencies(
  rclcpp
  nav_msgs
  std_msgs
  geometry_msgs
  ament_index_cpp
  OpenCV
)
install(TARGETS
  greedy_planner_node
  DESTINATION lib/${PROJECT_NAME}
)
ament_export_dependencies(
  rclcpp
  nav_msgs
  std_msgs
  geometry_msgs
  ament_index_cpp
  OpenCV
)

install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY maps
  DESTINATION share/${PROJECT_NAME}

)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()
