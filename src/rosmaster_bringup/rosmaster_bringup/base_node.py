#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
import math
import time


class OdomPublisher(Node):
    def __init__(self):
        super().__init__('base_node')

        # Parametros
        self.declare_parameter('wheelbase', 0.235)
        self.declare_parameter('linear_scale_x', 1.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_footprint_frame', 'base_footprint')
        self.declare_parameter('pub_odom_tf', True)

        self.wheelbase = self.get_parameter('wheelbase').value
        self.linear_scale_x = self.get_parameter('linear_scale_x').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_footprint_frame').value
        self.pub_tf = self.get_parameter('pub_odom_tf').value

        # Estado del robot
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.last_time = self.get_clock().now()

        # Comunicaciones
        self.vel_sub = self.create_subscription(
            Twist, 'vel_raw', self.handle_vel, 50)
        self.odom_pub = self.create_publisher(Odometry, 'odom_raw', 50)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('Nodo base_node iniciado')

    def handle_vel(self, msg: Twist):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time

        v = msg.linear.x * self.linear_scale_x
        steer_angle_deg = msg.linear.y
        steer_angle_rad = math.radians(steer_angle_deg)

        # Calculo de movimiento Ackermann
        if abs(steer_angle_rad) > 1e-4:
            turn_radius = self.wheelbase / math.tan(steer_angle_rad)
            angular_z = v / turn_radius
        else:
            angular_z = 0.0

        delta_heading = angular_z * dt
        delta_x = v * math.cos(self.heading) * dt
        delta_y = v * math.sin(self.heading) * dt

        self.x += delta_x
        self.y += delta_y
        self.heading += delta_heading

        # Cuaternion desde heading (yaw)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.heading)

        # Publicar Odometry
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = angular_z

        self.odom_pub.publish(odom)

        # Publicar TF
        if self.pub_tf:
            t = TransformStamped()
            t.header.stamp = current_time.to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
