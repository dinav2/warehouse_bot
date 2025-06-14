from setuptools import setup
import os

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/main_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='1461190907@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_yolo = vision.vision_yolo:main',
            'video_saver = vision.video_saver:main',
            'receptor_zmq_node = vision.receptor_zmq_node:main',
            'initial_pose_node = vision.initial_pose_node:main',

        ],
    },
)

