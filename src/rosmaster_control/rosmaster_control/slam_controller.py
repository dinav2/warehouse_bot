#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Int32
import numpy as np
from math import atan2, sin, cos, pi

class AckermannPathFollower(Node):
    def __init__(self):
        super().__init__('ackermann_path_follower')

        # Subscripciones y publicaciones
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.create_subscription(Int32, '/current_goal_index', self.goal_index_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.done_pub = self.create_publisher(Int32, '/goal_completed', 10)

        # Parámetros físicos del robot
        self.L = 0.235
        self.dt = 0.1
        self.max_steer = pi / 4
        self.max_velocity = 0.1
        self.min_velocity = 0.04
        self.tolerance = 0.1

        # Ganancias de control
        self.k1 = 1.0
        self.k2 = 1.5

        # Estados
        self.state = np.array([0.0, 0.0, 0.0])
        self.pose_received = False
        self.path = []
        self.current_goal_idx = 0
        self.goal_index = -1  # ID del objetivo actual desde el servidor

        self.timer = self.create_timer(self.dt, self.control_loop)

    def path_callback(self, msg: Path):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.current_goal_idx = 0
        self.get_logger().info(f"Trayectoria recibida con {len(self.path)} puntos.")

    def goal_index_callback(self, msg: Int32):
        self.goal_index = msg.data
        self.get_logger().info(f"ID del objetivo recibido: {self.goal_index}")

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        theta = atan2(siny_cosp, cosy_cosp)

        self.state = np.array([x, y, theta])
        self.pose_received = True

    def control_loop(self):
        if not self.pose_received or not self.path:
            return

        if self.current_goal_idx >= len(self.path):
            self.get_logger().info("Meta final alcanzada.")
            self.publish_stop()

            # Confirmar al servidor
            if self.goal_index >= 0:
                msg = Int32()
                msg.data = self.goal_index
                self.done_pub.publish(msg)
                self.get_logger().info(f"Confirmación enviada del objetivo {self.goal_index}")
                self.goal_index = -1  # Resetear

            return

        x, y, theta = self.state
        goal_x, goal_y = self.path[self.current_goal_idx]
        dx, dy = goal_x - x, goal_y - y
        dist = np.hypot(dx, dy)

        if dist < self.tolerance:
            self.current_goal_idx += 1
            return

        alpha = atan2(dy, dx)
        heading_error = self.wrap_to_pi(alpha - theta)

        if abs(heading_error) > pi / 2:
            direction = -1
            heading_error = self.wrap_to_pi(heading_error + pi)
        else:
            direction = 1

        lateral_error = -sin(theta - alpha) * dist
        steer = self.k2 * heading_error + self.k1 * atan2(lateral_error, 1.0)
        steer += -0.01
        steer = max(min(steer, self.max_steer), -self.max_steer)

        v = direction * min(self.max_velocity, max(self.min_velocity, dist))

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = steer
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def wrap_to_pi(self, angle):
        return (angle + pi) % (2 * pi) - pi

def main(args=None):
    rclpy.init(args=args)
    node = AckermannPathFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
