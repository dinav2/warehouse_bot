from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import os

def generate_launch_description():
    urdf_path = get_package_share_path('rosmaster_description')
    
    # Path to the URDF and RViz configuration files
    default_model_path = os.path.join(urdf_path, 'urdf', 'yahboomcar_R2.urdf')
    default_rviz_config_path = os.path.join(urdf_path, 'rviz', 'urdf.rviz')
    
    # Launch configuration variables
    gui = LaunchConfiguration('gui')
    urdf_model = LaunchConfiguration('urdf_model')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_simulator = LaunchConfiguration('use_simulator')
    use_rviz = LaunchConfiguration('use_rviz')
    
    # Declare the launch arguments
    declare_model_path_cmd = DeclareLaunchArgument(
        name='urdf_model',
        default_value=default_model_path,
        description='Absolute path to robot urdf file')
        
    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        name='rviz_config_file',
        default_value=default_rviz_config_path,
        description='Absolute path to rviz config file')
        
    declare_use_joint_state_publisher_gui_cmd = DeclareLaunchArgument(
        name='gui',
        default_value='True',
        description='Flag to enable joint_state_publisher_gui')
        
    declare_use_simulator_cmd = DeclareLaunchArgument(
        name='use_simulator',
        default_value='True',
        description='Flag to enable use of Gazebo')
        
    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='True',
        description='Flag to enable use of RViz')
    
    # Get URDF via xacro
    robot_description_content = Command(
        [
            'xacro ',
            urdf_model
        ]
    )
    
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=ParameterValue.PARAMETER_VALUE)}
    
    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_path('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
        condition=IfCondition(use_simulator)
    )
    
    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'robot'],
        condition=IfCondition(use_simulator),
        output='screen'
    )
    
    # Joint State Publisher
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=UnlessCondition(gui)
    )
    
    # Joint State Publisher GUI
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(gui)
    )
    
    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description]
    )
    
    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(use_rviz)
    )
    
    # Create and return launch description
    return LaunchDescription([
        declare_model_path_cmd,
        declare_rviz_config_file_cmd,
        declare_use_joint_state_publisher_gui_cmd,
        declare_use_simulator_cmd,
        declare_use_rviz_cmd,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        rviz_node
    ])