from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import os

def generate_launch_description():
    urdf_path = get_package_share_path('rosmaster_description')
    
    # Path to the URDF and RViz configuration files
    default_model_path = os.path.join(urdf_path, 'urdf', 'rosmaster.urdf.xacro')
    default_rviz_config_path = os.path.join(urdf_path, 'rviz', 'urdf.rviz')
    
    # Launch configuration variables
    gui = LaunchConfiguration('gui')
    urdf_model = LaunchConfiguration('urdf_model')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_simulator = LaunchConfiguration('use_simulator')
    use_rviz = LaunchConfiguration('use_rviz')

    print(f"URDF: {default_model_path}") 
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
            default_model_path
        ]
    )
    
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}
    
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
        arguments=['-topic', 'robot_description', '-entity', 'robot','-x','-0.8','-y','-0.9','-z','0.0255'],
        condition=IfCondition(use_simulator),
        output='screen'
    )

    # Spawn model
    spawn_warehouse = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-file', '/root/tsuru_ws/src/rosmaster_simulation/models/warehouse1/model.sdf', '-entity', 'warehouse'],
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
        parameters=[robot_description, {'use_sim_time': True}]
    )
    

    # Controller Manager
    control = Node(
        package='controller_manager',
	executable='ros2_control_node',
	parameters=[
	    robot_description,
	    {'use_sim_time': True},
	    os.path.join(
		get_package_share_path('rosmaster_description'),
		'params',
		'joint_controller.yaml'
	    )
	],
	output='both'
    )

    # Load Controllers
    load_controllers = [
	TimerAction(
            period=50.0,
	    actions=[
	        Node(
	            package='controller_manager',
	            executable='spawner.py',
	            arguments=['ackermann_steering_controller', '-c', '/controller_manager'],
	        ),
	        Node(
	            package='controller_manager',
	            executable='spawner.py',
	            arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
	        ),
	    ]
	)
    ]


    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
	parameters=[{'use_sim_time': True}],
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
	spawn_warehouse,
	*load_controllers,
        rviz_node
    ])
