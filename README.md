# Warehouse Bot - Autonomous Navigation and Signal Detection Robot

## Project Overview

The Warehouse Bot is a sophisticated autonomous mobile robot built on ROS2 (Robot Operating System 2) designed for warehouse operations. It combines advanced navigation, computer vision, and path planning capabilities to autonomously navigate through warehouse environments while detecting and classifying visual signals.

## Key Features

- **Autonomous Navigation**: SLAM-based mapping and localization with Nav2 navigation stack
- **Computer Vision**: Real-time signal detection and classification using TensorRT-optimized deep learning
- **SLAM Mapping**: Real-time mapping using slam_toolbox with LIDAR data
- **AMCL Localization**: Adaptive Monte Carlo Localization for precise positioning
- **Ackermann Steering Control**: Advanced vehicle dynamics for smooth navigation
- **Multi-Phase Mission Planning**: Intelligent goal selection and execution
- **Sensor Fusion**: IMU, LIDAR, and camera data integration with EKF
- **ROS2 Native**: Built with modern ROS2 architecture for scalability and performance

## System Architecture

### Core Components

1. **Robot Control System** (`rosmaster_bringup`)
   - Ackermann steering driver
   - Base odometry and kinematics
   - Sensor data publishing (IMU, encoders)

2. **Path Planning System** (`planner`)
   - Greedy path planner with obstacle avoidance
   - Hybrid A* algorithm implementation
   - Mission planner with multi-phase execution

3. **Computer Vision System** (`vision`)
   - TensorRT-based signal classification
   - ArUco marker detection and pose estimation
   - ZMQ-based communication for real-time processing

4. **Robot Description** (`rosmaster_description`)
   - URDF/Xacro robot model
   - Visual and collision meshes

5. **Localization System** (`rosmaster_localization`)
   - AMCL configuration and parameters
   - Map server setup and configuration
   - Lifecycle management for localization nodes

6. **Simulation Environment** (`rosmaster_simulation`)
   - Gazebo simulation support

## Hardware

- **Base Platform**: Ackermann steering robot chassis
- **Sensors**:
  - RPLIDAR A2/A1 for 2D laser scanning
  - IMU for orientation sensing
  - RGB-D Camera (Astra Pro Plus) for vision
  - Wheel encoders for odometry
- **Computing**: 
  - NVIDIA GPU for TensorRT inference
  - Linux system with ROS2 support

## Software Dependencies

### ROS2 Packages
- `rclpy` - Python ROS2 client library
- `geometry_msgs` - Geometry message types
- `nav_msgs` - Navigation message types
- `tf2_ros` - Transform library
- `robot_state_publisher` - Robot state publishing
- `joint_state_publisher` - Joint state management
- `robot_localization` - EKF-based sensor fusion
- `imu_filter_madgwick` - IMU orientation filtering
- `slam_toolbox` - SLAM mapping and localization
- `nav2_map_server` - Map serving for navigation
- `nav2_amcl` - Adaptive Monte Carlo Localization
- `nav2_lifecycle_manager` - Node lifecycle management
- `nav2_bringup` - Navigation stack launch files

### Computer Vision
- `OpenCV` - Computer vision library
- `TensorRT` - NVIDIA inference optimization
- `PyCUDA` - CUDA Python bindings
- `ZMQ` - Zero Message Queue for IPC

### Additional Dependencies
- `ament_cmake` - ROS2 build system
- `colcon` - Build tool
- `transforms3d` - 3D transformations

## Installation

### 1. Prerequisites
Ensure you have ROS2 (Humble/Foxy) installed and the following packages:
```bash
sudo apt install ros-$ROS_DISTRO-slam-toolbox
sudo apt install ros-$ROS_DISTRO-nav2-bringup
sudo apt install ros-$ROS_DISTRO-robot-localization
sudo apt install ros-$ROS_DISTRO-imu-filter-madgwick
```

### 2. Clone the Repository
```bash
git clone <repository_url> warehouse_bot
cd warehouse_bot
```

### 3. Install ROS2 Dependencies
```bash
cd src
rosdep install --from-paths . --ignore-src -r -y
```

### 4. Build the Workspace
```bash
cd ..
colcon build --symlink-install
source install/setup.bash
```

### 5. Hardware Setup
- Connect LIDAR to appropriate USB port
- Connect camera to USB 3.0 port
- Ensure robot base is properly connected
- Calibrate camera using provided calibration script:
```bash
python3 src/vision/vision/calibrateCamera.py
```

## Usage

### Basic Launch
Launch the complete system (localization mode with pre-built map):
```bash
ros2 launch rosmaster_bringup bringup.launch.py
```

### SLAM Mode
For mapping new environments (includes SLAM + Navigation):
```bash
ros2 launch rosmaster_bringup slam.launch.py
```

## System Topics and Services

### Key Topics
- `/cmd_vel` - Velocity commands (geometry_msgs/Twist)
- `/odom` - Odometry data (nav_msgs/Odometry)
- `/scan` - LIDAR data (sensor_msgs/LaserScan)
- `/imu/data_raw` - Raw IMU data (sensor_msgs/Imu)
- `/planned_path` - Planned path (nav_msgs/Path)
- `/mission_status` - Mission status updates (std_msgs/String)
- `/current_goal_index` - Current goal index (std_msgs/Int32)

## AI and Computer Vision

### Signal Detection
- Uses custom-trained deep learning model optimized with TensorRT
- Real-time inference at 30 FPS
- Supports multiple signal classes with configurable confidence thresholds

### SLAM and Localization
- **SLAM Toolbox**: Real-time simultaneous localization and mapping
- **AMCL**: Adaptive Monte Carlo Localization for robot pose estimation
- **Map Server**: Serves pre-built maps for navigation in known environments
- **Lifecycle Manager**: Manages the lifecycle of navigation nodes

### Path Planning
- **Custom Mission Planner**: Multi-phase goal execution with signal-based decision making
- **Greedy Planner**: Fast path planning with obstacle avoidance
- **Hybrid A***: Advanced path planning algorithm for complex environments
- **Obstacle Avoidance**: Static obstacle detection and avoidance using inflated occupancy grids

## Configuration

### Robot Parameters
Key parameters can be modified in launch files:
- `wheelbase`: Ackermann steering wheelbase (default: 0.235m)
- `linear_scale_x`: Linear velocity scaling factor
- `pub_odom_tf`: Enable/disable odometry transform publishing

### Vision Parameters
- `CONF_THRESHOLD`: Detection confidence threshold (default: 0.0001)
- `IOU_THRESHOLD`: Non-maximum suppression threshold (default: 0.5)
- Camera calibration stored in `calibracion.npz`

### Planning Parameters
- `robot_radius`: Robot safety radius for obstacle avoidance
- `map_scale_factor`: Map resolution scaling (default: 10.0)
- Goal positions defined in planning node

## 🗂️ Project Structure

```
warehouse_bot/
├── src/
│   ├── rosmaster_bringup/        # Main system launch and control
│   ├── rosmaster_control/        # Robot control interfaces
│   ├── rosmaster_description/    # Robot URDF models
│   ├── rosmaster_simulation/     # Gazebo simulation
│   ├── rosmaster_localization/   # Localization and AMCL configuration
│   ├── planner/                  # Path planning algorithms
│   ├── vision/                   # Computer vision system
├── calibracion.npz               # Camera calibration data
└── README.md                     # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check USB 3.0 connection
   - Verify camera permissions: `sudo chmod 666 /dev/video*`

2. **LIDAR connection failed**
   - Check USB connection and permissions
   - Verify device path: `ls /dev/ttyUSB*`

3. **TensorRT model loading error**
   - Ensure CUDA and TensorRT are properly installed
   - Check GPU memory availability
   - Verify model path: `/root/modelos/signals.engine`

4. **Navigation issues**
   - Verify map file exists: `src/planner/maps/mapaFidedigno1.png`
   - Check AMCL pose initialization
   - Ensure proper coordinate frame transformations

### Debug Tools
```bash
# Check all active topics
ros2 topic list

# Monitor robot odometry
ros2 topic echo /odom

# Visualize in RViz
ros2 run rviz2 rviz2

# Check TF tree
ros2 run tf2_tools view_frames
```

## 📋 TODO / Future Enhancements

- [ ] Add multi-robot coordination capabilities
- [ ] Implement advanced obstacle avoidance algorithms
- [ ] Develop web-based monitoring dashboard
- [ ] Integration with warehouse management systems
- [ ] Add support for dynamic obstacle detection


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.