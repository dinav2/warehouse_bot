import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import Pose2D

import zmq
import json
import threading
import math

class ZMQSubscriberNode(Node):
    def __init__(self):
        super().__init__('zmq_subscriber_node')
        self.clase_pub = self.create_publisher(Int32, '/clase_detectada', 10)
        self.pose_pub = self.create_publisher(Pose2D, '/pose_robot', 10)

        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:6001")  # Cambia si es remoto
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        thread = threading.Thread(target=self.zmq_listen, daemon=True)
        thread.start()

    def zmq_listen(self):
        while rclpy.ok():
            try:
                msg = self.socket.recv_string()
                data = json.loads(msg)

                clase = data.get("clase")
                clase_msg = Int32()
                if clase is not None and clase != "None":
                    clase_msg.data = int(clase)
                else:
                    clase_msg.data = -1  # valor para "no clase"
                self.clase_pub.publish(clase_msg)

                pose_msg = Pose2D()
                pose_msg.x = float(data.get("x") or 0.0)
                pose_msg.y = float(data.get("y") or 0.0)
                yaw_rad = float(data.get("yaw") or 0.0)
                pose_msg.theta = math.degrees(yaw_rad)

                self.pose_pub.publish(pose_msg)

            except Exception as e:
                self.get_logger().error(f"Error al recibir o publicar mensaje ZMQ: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ZMQSubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

