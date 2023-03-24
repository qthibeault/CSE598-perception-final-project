from typing import Any, Callable, Optional, Protocol, TypedDict, overload

from torch import FloatTensor, Int64Tensor, Tensor, UInt8Tensor, Weights

class _Transforms(Protocol):
    @overload
    def __call__(self, img: list[object]) -> list[Tensor]: ...
    @overload
    def __call__(self, img: object) -> Tensor: ...

class MaskRCNN_ResNet50_FPN_V2_Weights:
    DEFAULT: Weights
    COCO_V1: Weights
    transforms: Callable[[Any], Tensor]

class _Output(TypedDict):
    boxes: FloatTensor
    labels: Int64Tensor
    scores: Tensor
    masks: UInt8Tensor

class MaskRCNN:
    def eval(self): ...
    def __call__(self, images: list[Tensor] | Tensor) -> list[_Output]: ...

def maskrcnn_resnet50_fpn_v2(weights: Optional[Weights] = ..., progress: bool = ...) -> MaskRCNN: ...
