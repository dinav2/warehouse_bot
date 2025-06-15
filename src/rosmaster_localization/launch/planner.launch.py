from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Rutas de paquetes
    rosmaster_planner = get_package_share_directory('rosmaster_planner')
    sllidar_ros2 = get_package_share_directory('sllidar_ros2')

    # Archivos
    map_yaml_file = os.path.join(rosmaster_planner, 'maps', 'map2.yaml')
    amcl_params_file = os.path.join(rosmaster_planner, 'params', 'amcl_params.yaml')

    # Verificar existencia del archivo .pgm
    map_image_path = os.path.join(rosmaster_planner, 'maps', 'map.pgm')
    if not os.path.exists(map_image_path):
        print(f"[ERROR] El archivo de imagen del mapa no existe: {map_image_path}")

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),

        # LIDAR SL-LiDAR A1
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sllidar_ros2, 'launch', 'sllidar_launch.py')
            ),
            launch_arguments={
                'serial_port': '/dev/rplidar',
                'frame_id': 'laser_link',
                'use_sim_time': use_sim_time
            }.items()
        ),

        # Nodo map_server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'yaml_filename': map_yaml_file}
            ]
        ),

        # Nodo amcl
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                amcl_params_file
            ]
        ),

        # Lifecycle Manager para map_server y amcl
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart': True},
                {'node_names': ['map_server', 'amcl']}
            ]
        )
    ])
