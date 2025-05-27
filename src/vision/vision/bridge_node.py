import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import zmq
import numpy as np

class ROSBridgeNode(Node):
    def __init__(self):
        super().__init__('ros_bridge_node')
        self.get_logger().info("Nodo ROS 2 conectado al motor de inferencia")

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Image, '/yolo/detections/image', 10)

        # ZeroMQ socket (client)
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, encoded = cv2.imencode(".jpg", frame)
            self.socket.send(encoded.tobytes())

            jpg_response = self.socket.recv()
            annotated = cv2.imdecode(np.frombuffer(jpg_response, np.uint8), cv2.IMREAD_COLOR)
            ros_image = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self.pub.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f"Error en procesamiento de imagen: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ROSBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
