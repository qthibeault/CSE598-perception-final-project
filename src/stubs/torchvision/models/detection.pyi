from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, TypedDict, overload

from torch import FloatTensor, Int64Tensor, Tensor, UInt8Tensor, Weights

class _Transforms(Protocol):
    @overload
    def __call__(self, img: list[object]) -> list[Tensor]: ...
    @overload
    def __call__(self, img: object) -> Tensor: ...


class MaskRCNN_ResNet50_FPN_V2_Weights(Enum, Weights):
    DEFAULT = ...
    COCO_V1 = ...

    def transforms(self) -> Callable[[Any], Tensor]: ...

if TYPE_CHECKING:
    class MaskRCNNResult(TypedDict):
        boxes: FloatTensor
        labels: Int64Tensor
        scores: Tensor
        masks: UInt8Tensor

class MaskRCNN:
    def eval(self): ...
    def __call__(self, images: list[Tensor] | Tensor) -> list[MaskRCNNResult]: ...

def maskrcnn_resnet50_fpn_v2(
    weights: Optional[Weights] = ...,
    progress: bool = ...,
) -> MaskRCNN: ...
