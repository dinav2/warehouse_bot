import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rosmaster_calibration',
            executable='auto_calibration',
            name='auto_calibration',
            output='screen',
            parameters=[{
                'forward_distance': 0.6,
                'velocity': 0.12,
                'covariance_threshold': 0.04,
                'required_stable_samples': 5,
                'auto_start': True,
            }]
        )
    ]) 