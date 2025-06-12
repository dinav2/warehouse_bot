from setuptools import setup

package_name = 'rosmaster_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/auto_calibration.launch.py']),
    ],
    install_requires=['setuptools', 'rclpy', 'numpy'],
    zip_safe=True,
    maintainer='dinav2',
    maintainer_email='wealthydina@icloud.com',
    description='Node that performs an initial auto-calibration manoeuvre (forward/backward) until SLAM covariance converges, then publishes /initialpose.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_calibration = rosmaster_calibration.auto_calibration:main',
        ],
    },
) 