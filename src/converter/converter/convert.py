import cv2
import cv_bridge
import rclpy
import rclpy.node as node
import sensor_msgs.msg as msg


class Converter(node.Node):
    def __init__(self):
        super().__init__("converter", parameter_overrides=[])
        self.declare_parameter("video")

        self._video_name = self.get_parameter("video").get_parameter_value().string_value
        self._video = cv2.VideoCapture(self._video_name)

        if not self._video.isOpened():
            raise RuntimeError(f"Could not open video {self._video_name}")

        self._bridge = cv_bridge.CvBridge()
        self._image_topic = self.create_publisher(msg.Image, "images", 10)
        self._publish_timer = self.create_timer(0.1, self._send_image)

    def _send_image(self):
        # Retrieve the next frame from the video
        read_success, frame = self._video.read()

        # Stop if there are no frames left to read
        if not read_success:
            # Close video file and stop publish timer
            self._video.release()
            self._publish_timer.cancel()

        # Convert the OpenCV Mat to a ROS2 Image
        msg = self._bridge.cv2_to_imgmsg(frame)

        # Publish frame to topic
        self._image_topic.publish(msg)


def main(args: list[str] | None = None):
    rclpy.init(args=args)
    rclpy.spin(Converter())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
