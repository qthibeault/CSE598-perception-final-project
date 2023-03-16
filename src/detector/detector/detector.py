from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Detector(Node):
    def __init__(self) -> None:
        super().__init__("detector")
        self._publisher = self.create_publisher(String, "topic", 10)
        self._subscription = self.create_subscription(String, "topic", self.on_message, 10)
        self.get_logger().info("--- [ Detector node online ]")

    def on_message(self, msg: String) -> None:
        pass


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    rclpy.spin(Detector())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
