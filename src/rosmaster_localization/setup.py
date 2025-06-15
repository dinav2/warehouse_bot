from setuptools import setup
import os
from glob import glob

package_name = 'rosmaster_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*launch.py'))),
	(os.path.join('share',package_name,'maps'),glob(os.path.join('maps','*.*'))),
	(os.path.join('share',package_name,'params'),glob(os.path.join('params','*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='wealthydina@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = rosmaster_localization.planner_node:main',
            'mission_planner = rosmaster_localization.mission_planner:main',
	    'path_planner = rosmaster_localization.path_planner:main',
            'visualizer = rosmaster_localization.visualizer:main',
        ],
    },
)
