from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Rutas de paquetes
    rosmaster_bringup = get_package_share_directory('rosmaster_bringup')
    slam_toolbox = get_package_share_directory('slam_toolbox')
    nav2_bringup = get_package_share_directory('nav2_bringup')
    sllidar_ros2 = get_package_share_directory('sllidar_ros2')

    # 1. Rosmaster Bringup
    bringup_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rosmaster_bringup, 'launch', 'bringup.launch.py')
        )
    )

    # 2. LiDAR
    lidar_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sllidar_ros2, 'launch', 'sllidar_launch.py')
        ),
        launch_arguments={
            'serial_port': '/dev/rplidar',
            'frame_id': 'laser_link',
            'use_sim_time': use_sim_time
        }.items()
    )

    # 3. SLAM Toolbox (retrasado 5 segundos)
    slam_node = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_toolbox, 'launch', 'online_async_launch.py')
                ),
                launch_arguments={
                    'slam_params_file': os.path.join(rosmaster_bringup, 'params', 'mapper_params_online_async'),
                    'use_sim_time': use_sim_time
                }.items()
            )
        ]
    )

    # 4. Nav2 (retrasado 10 segundos)
    nav2_node = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup, 'launch', 'navigation_launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'autostart': 'true',
                    'params_file': os.path.join(rosmaster_bringup, 'params', 'nav2_params.yaml')
                }.items()
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        bringup_node,
        lidar_node,
        slam_node,
        nav2_node
    ])
