#!/usr/bin/env python3
"""
AutoCalibrationManager
----------------------
Nodo ROS 2 que realiza una maniobra lineal adelante-atrás para que el algoritmo
SLAM (p.ej. AMCL / SLAM Toolbox) converja. Vigila la varianza de la pose y la
distancia a obstáculos; cuando la estimación es fiable publica /initialpose y
avisa al resto del sistema mediante el tópico /calibration/pose_ready (std_msgs/Bool).

El robot utiliza transmisión Ackermann, pero para simplificar usamos /cmd_vel con
velocidades lineales ±v y angular = 0. Si fuese necesario, puede adaptarse a
ackermann_msgs/AckermannDrive.
"""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

# Al publicar la pose inicial usamos el mismo mensaje que recibimos
def pose_msg_copy(src: PoseWithCovarianceStamped) -> PoseWithCovarianceStamped:
    dst = PoseWithCovarianceStamped()
    dst.header = src.header
    dst.pose = src.pose
    return dst


class State(Enum):
    IDLE = 0
    MOVING_FWD = 1
    MOVING_BWD = 2
    EVALUATE = 3
    FINISHED = 4


class AutoCalibrationManager(Node):
    def __init__(self):
        super().__init__('auto_calibration_manager')

        # ---------------- Parameters ----------------
        self.declare_parameter('forward_distance', 0.5)  # m
        self.declare_parameter('velocity', 0.12)  # m/s
        self.declare_parameter('covariance_threshold', 0.05)
        self.declare_parameter('required_stable_samples', 5)
        self.declare_parameter('obstacle_distance_threshold', 0.4)  # m
        self.declare_parameter('laser_topic', '/scan')
        self.declare_parameter('pose_topic', '/amcl_pose')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('auto_start', True)

        # ---------------- Internal vars ----------------
        self.state: State = State.IDLE
        self._last_pose_msg: Optional[PoseWithCovarianceStamped] = None
        self._stable_counter: int = 0
        self._move_start_time: Optional[float] = None
        self._move_duration: float = 0.0
        self._dt: float = 0.05  # control timestep

        # Obstacle monitoring
        self._ahead_clear: bool = False
        self._ahead_distance: float = float('inf')

        # ---------------- ROS entities ----------------
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        laser_topic = self.get_parameter('laser_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value

        self.create_subscription(PoseWithCovarianceStamped, pose_topic, self.on_pose, 10)
        self.create_subscription(LaserScan, laser_topic, self.on_scan, 10)

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.ready_pub = self.create_publisher(Bool, '/calibration/pose_ready', 10)
        # Also publish initialpose for compatibility with nav stack
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.timer = self.create_timer(self._dt, self.control_loop)

        if self.get_parameter('auto_start').get_parameter_value().bool_value:
            self.start_calibration()

    # ---------------- Callbacks ----------------
    def on_pose(self, msg: PoseWithCovarianceStamped):
        self._last_pose_msg = msg

        if self.state in (State.MOVING_FWD, State.MOVING_BWD, State.EVALUATE):
            if self.is_cov_good(msg):
                self._stable_counter += 1
            else:
                self._stable_counter = 0

    def on_scan(self, msg: LaserScan):
        # Evaluate minimum range ahead (we consider ±10° sector)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = np.array(msg.ranges)

        # sector around 0 rad ± ~0.17 rad (~10°)
        forward_mask = np.abs(
            angle_min + np.arange(len(ranges)) * angle_increment) < 0.17
        forward_ranges = ranges[forward_mask]
        ahead_distance = np.nanmin(forward_ranges) if forward_ranges.size else float('inf')
        self._ahead_distance = ahead_distance
        thresh = self.get_parameter('obstacle_distance_threshold').get_parameter_value().double_value
        self._ahead_clear = bool(ahead_distance > thresh)

    # ---------------- Helper methods ----------------
    def is_cov_good(self, msg: PoseWithCovarianceStamped) -> bool:
        cov = np.array(msg.pose.covariance).reshape(6, 6)
        thresh = self.get_parameter('covariance_threshold').get_parameter_value().double_value
        xy_cov_ok = cov[0, 0] < thresh and cov[1, 1] < thresh
        yaw_idx = 5
        yaw_cov_ok = cov[yaw_idx, yaw_idx] < thresh
        return xy_cov_ok and yaw_cov_ok

    def publish_twist(self, vx: float):
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def start_calibration(self):
        # Ensure obstacles clear: wait for LiDAR data first if not clear.
        if not self._ahead_clear:
            self.get_logger().warn('Obstacle detected at %.2f m, waiting to start calibration…' % self._ahead_distance)
            return  # stay in IDLE until path is clear
        fwd_dist = self.get_parameter('forward_distance').get_parameter_value().double_value
        vel = self.get_parameter('velocity').get_parameter_value().double_value
        self._move_duration = fwd_dist / vel if vel > 0 else 0.0
        self._move_start_time = None
        self.state = State.MOVING_FWD
        self.get_logger().info('⏩ Starting auto-calibration: moving forward %.2f m', fwd_dist)

    def stop_robot(self):
        self.publish_twist(0.0)

    # ---------------- Main FSM loop ----------------
    def control_loop(self):
        now = self.get_clock().now().seconds_nanoseconds()[0] + \
            self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        if self.state == State.IDLE:
            return

        if self.state == State.MOVING_FWD:
            # safety check during motion
            if not self._ahead_clear:
                self.get_logger().error('🚨 Obstacle appeared at %.2f m. Aborting calibration.' % self._ahead_distance)
                self.stop_robot()
                self.state = State.IDLE
                return
            if self._move_start_time is None:
                # first iteration
                self._move_start_time = now
            elapsed = now - self._move_start_time
            if elapsed < self._move_duration:
                self.publish_twist(self.get_parameter('velocity').get_parameter_value().double_value)
            else:
                # finished forward
                self.stop_robot()
                self.state = State.MOVING_BWD
                self._move_start_time = now
                self.get_logger().info('⏪ Moving backward same distance')

        elif self.state == State.MOVING_BWD:
            # When moving backward use same forward cone? robot orientation unchanged; assume safe.
            elapsed = now - (self._move_start_time or now)
            if elapsed < self._move_duration:
                self.publish_twist(-self.get_parameter('velocity').get_parameter_value().double_value)
            else:
                self.stop_robot()
                self.state = State.EVALUATE
                self.get_logger().info('⏹ Movement finished. Evaluating covariance …')

        elif self.state == State.EVALUATE:
            # Wait for stable covariance
            required_samples = int(self.get_parameter('required_stable_samples').get_parameter_value().integer_value)
            if self._stable_counter >= required_samples and self._last_pose_msg:
                self.finish_calibration()

        # Nothing to do for FINISHED

    def finish_calibration(self):
        if not self._last_pose_msg:
            self.get_logger().error('No pose message to publish as initialpose')
            return
        self.stop_robot()
        pose_msg = pose_msg_copy(self._last_pose_msg)
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.init_pose_pub.publish(pose_msg)

        ready_msg = Bool()
        ready_msg.data = True
        self.ready_pub.publish(ready_msg)

        self.state = State.FINISHED
        self.get_logger().info('✅ Calibration finished. Initial pose published.')


# ---------------- Entry point ----------------

def main(args: list[str] | None = None):
    rclpy.init(args=args)
    node = AutoCalibrationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 