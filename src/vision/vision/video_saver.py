import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoSaver(Node):
    def __init__(self):
        super().__init__('video_saver')

        # Inicializa el puente entre ROS y OpenCV
        self.bridge = CvBridge()

        # Suscripción al tópico de imágenes
        self.sub_img = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.handle_image,
            10
        )

        # Inicializa variables de video
        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.output_file = 'video_output.mkv'
        self.fps = 30  # Puedes cambiarlo según tu fuente
        self.frame_size = None
        self.first_frame = True

        self.get_logger().info('VideoSaver iniciado y escuchando /image_raw')

    def handle_image(self, msg):
        try:
            # Convertir la imagen ROS a OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Inicializar VideoWriter si es la primera imagen
            if self.first_frame:
                self.frame_size = (frame.shape[1], frame.shape[0])
                self.video_writer = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, self.frame_size)
                self.first_frame = False
                self.get_logger().info(f'VideoWriter inicializado: {self.output_file}')

            # Escribir frame en el archivo de video
            self.video_writer.write(frame)

        except Exception as e:
            self.get_logger().error(f'Error al procesar imagen: {e}')

    def destroy_node(self):
        # Cerrar el archivo de video correctamente al finalizar
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info('Video guardado exitosamente.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
