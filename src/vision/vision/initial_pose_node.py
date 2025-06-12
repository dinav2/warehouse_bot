import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped
from std_srvs.srv import Trigger
import math

class InitialPoseService(Node):
    def __init__(self):
        super().__init__('initial_pose_service')

        self.latest_pose = None  # Guarda la última pose válida

        self.subscription = self.create_subscription(
            Pose2D,
            '/pose_robot',
            self.pose_callback,
            10
        )

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.srv = self.create_service(
            Trigger,
            '/set_initial_pose',
            self.handle_set_initial_pose
        )

        self.get_logger().info('Escuchando /pose_robot. Servicio activo en /set_initial_pose')

    def pose_callback(self, msg: Pose2D):
        self.latest_pose = msg  # Actualizar la última pose
        self.get_logger().debug(f'Pose recibida: x={msg.x:.2f}, y={msg.y:.2f}, θ={msg.theta:.2f}')

    def handle_set_initial_pose(self, request, response):
        if self.latest_pose is None:
            response.success = False
            response.message = 'No se ha recibido ninguna pose aún.'
            return response

        yaw = math.radians(self.latest_pose.theta) if abs(self.latest_pose.theta) > math.pi else self.latest_pose.theta
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose.position.x = self.latest_pose.x
        pose_msg.pose.pose.position.y = self.latest_pose.y
        pose_msg.pose.pose.orientation.z = qz
        pose_msg.pose.pose.orientation.w = qw

        pose_msg.pose.covariance = [0.0]*36
        pose_msg.pose.covariance[0] = 0.25
        pose_msg.pose.covariance[7] = 0.25
        pose_msg.pose.covariance[35] = 0.0685

        self.publisher.publish(pose_msg)
        self.get_logger().info(f'📍 Pose inicial publicada: x={self.latest_pose.x:.2f}, y={self.latest_pose.y:.2f}, yaw={math.degrees(yaw):.1f}°')

        response.success = True
        response.message = 'Pose publicada correctamente.'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = InitialPoseService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
