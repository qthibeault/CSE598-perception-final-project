from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

TARGET_LABEL = ""


class Detector(Node):
    def __init__(self):
        super().__init__("detector", parameter_overrides=[])

        self._bridge = CvBridge()
        self._model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self._x_publisher = self.create_publisher(Image, "object/pos_x", 10)
        self._y_publisher = self.create_publisher(Image, "object/pos_y", 10)
        self._subscription = self.create_subscription(Image, "camera", self.on_message, 10)
        self._last_frame = None

        self._model.eval()
        self.get_logger().info("--- [ Detector node online ]")

    def on_message(self, msg: Image):
        image = self._bridge.imgmsg_to_cv2(msg)
        x_pos, y_pos = self._identify(image)
        x_msg = Float64()
        x_msg.data = x_pos

        self._x_publisher.publish(x_msg)

        y_msg = Float64()
        y_msg.data = y_pos

        self._y_publisher.publish(y_msg)

    def _identify(self, image: cv2.Mat) -> tuple[float, float]:
        batch = MaskRCNN_ResNet50_FPN_V2_Weights.transforms(image).unsqueeze(0)  # type: ignore
        results = self._model(batch)

        for result in results:
            pass

        # TODO: Make sure the object is the same between frames

        return 0, 0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    rclpy.spin(Detector())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
