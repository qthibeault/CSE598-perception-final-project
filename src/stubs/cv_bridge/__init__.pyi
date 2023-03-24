from typing import Literal, Optional

from cv2 import Mat
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class CvBridgeError(TypeError): ...

class CvBridge:
    def imgmsg_to_cv2(
        self,
        img_msg: Image,
        desired_encoding: Literal["passthrough"] = ...,
    ) -> Mat: ...
    def cv2_to_imgmsg(
        self,
        cvim: Mat,
        encoding: Literal["passthrough"] = ...,
        header: Optional[Header] = ...,
    ) -> Image: ...
