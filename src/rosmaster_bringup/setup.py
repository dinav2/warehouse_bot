from setuptools import setup
import os
from glob import glob

package_name = 'rosmaster_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*launch.py'))),
	(os.path.join('share',package_name,'params'),glob(os.path.join('params', '*.yaml'))),
    ],
    install_requires=['setuptools','transforms3d'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='wealthydina@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'Ackermann_driver = rosmaster_bringup.Ackermann_driver:main',
		'base_node = rosmaster_bringup.base_node:main',
        ],
    },
)
