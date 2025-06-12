from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import AnyLaunchDescriptionSource

def generate_launch_description():
    astra_launch_path = '/root/yahboomcar_ros2_ws/software/library_ws/install/astra_camera/share/astra_camera/launch/astro_pro_plus.launch.xml'

    return LaunchDescription([
        # Incluir el launch XML de astra_camera
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(astra_launch_path)
        ),

        # Ejecutar inferenia_zmq.py dentro del entorno virtual
        ExecuteProcess(
            cmd=['bash', '-c', 'source /opt/venv_py36/bin/activate && python3 /root/tsuru_ws/src/vision/vision/inferenia_zmq.py'],
            output='screen'
        ),

        # Ejecutar el nodo receptor_zmq_node normal ROS 2
        Node(
            package='vision',
            executable='receptor_zmq_node',
            name='receptor_zmq_node'
        ),

        # Ejecutar bridge_node.py (script Python normal)
        ExecuteProcess(
            cmd=['python3', '/root/tsuru_ws/src/vision/vision/bridge_node.py'],
            output='screen'
        ),
    ])

