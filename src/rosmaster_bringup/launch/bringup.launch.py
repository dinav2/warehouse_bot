from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import os
from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    urdf_path = get_package_share_path('rosmaster_description')
    default_model_path = urdf_path / 'urdf/rosmaster.urdf.xacro'
    default_rviz_config_path = urdf_path / 'rviz/yahboomcar.rviz'

    gui_arg = DeclareLaunchArgument(name='gui', default_value='false', choices=['true', 'false'],
                                    description='Flag to enable joint_state_publisher_gui')
    model_arg = DeclareLaunchArgument(name='model', default_value=str(default_model_path),
                                      description='Absolute path to robot urdf file')
    rviz_arg = DeclareLaunchArgument(name='rvizconfig', default_value=str(default_rviz_config_path),
                                     description='Absolute path to rviz config file')
    pub_odom_tf_arg = DeclareLaunchArgument('pub_odom_tf', default_value='false',
                                            description='Whether to publish the tf from the original odom to the base_footprint')

    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]),
                                       value_type=str)

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    # Depending on gui parameter, either launch joint_state_publisher or joint_state_publisher_gui
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        condition=UnlessCondition(LaunchConfiguration('gui'))
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    imu_filter_config = os.path.join(
        get_package_share_directory('rosmaster_bringup'),
        'params',
        'imu_filter_param.yaml'
    )

    driver_node = Node(
        package='rosmaster_bringup',
        executable='Ackermann_driver',
    )

    base_node = Node(
        package='rosmaster_bringup',
        executable='base_node',
        parameters=[{'pub_odom_tf': LaunchConfiguration('pub_odom_tf')}]
    )

    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        parameters=[imu_filter_config]
    )

    ekf_config = os.path.join(
        get_package_share_directory('rosmaster_bringup'),
	'params',
	'ekf.yaml'
    )

    ekf_node = Node(
	package='robot_localization',
	executable='ekf_node',
	name='ekf_filter_node',
	output='screen',
	parameters=[ekf_config],
	remappings=[
	    ('/odometry/filtered', 'odom')
	]
    )

    yahboom_joy_node = Node(
        package='yahboomcar_ctrl',
        executable='yahboom_joy_R2',
    )

    vision_launch_path = os.path.join(
        get_package_share_directory('vision'),
        'launch',
        'main_launch.py'
    )

    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(vision_launch_path)
    )

    initial_pose_node = Node(
        package='vision',
        executable='initial_pose_node',
    )

    planner_launch_path = os.path.join(
        get_package_share_directory('rosmaster_planner'),
        'launch',
        'planner.launch.py'
    )

    planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(planner_launch_path)
    )

    return LaunchDescription([
        gui_arg,
        model_arg,
        rviz_arg,
        pub_odom_tf_arg,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        robot_state_publisher_node,
        #rviz_node,
        vision_launch,
        driver_node,
        base_node,
        imu_filter_node,
        ekf_node,
        initial_pose_node,
        yahboom_joy_node,
        planner_launch
    ])
