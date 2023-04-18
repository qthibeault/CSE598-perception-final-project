from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Protocol, Type
from uuid import UUID, uuid4

import click
import cv2
import torchvision.models.detection as models

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BBox:
    """The bounding box of an object."""

    x: tuple[float, float] = field()
    y: tuple[float, float] = field()

    def __post_init__(self):
        if self.x[0] >= self.x[1]:
            raise ValueError("x0 must be strictly less than x1")

        if self.y[0] >= self.y[1]:
            raise ValueError("y0 must be strictly less than y1")

    @property
    def width(self) -> float:
        return self.x[1] - self.x[0]

    @property
    def height(self) -> float:
        return self.y[1] - self.y[0]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        x_center = (self.x[1] - self.x[0]) / 2
        y_center = (self.y[1] - self.y[0]) / 2

        return Point(x_center, y_center)

    def iou(self, other: BBox) -> float:
        """Compute the Intersection-over-Union (IOU) score between two bounding boxes."""

        x1 = max(self.x[0], other.x[0])
        x2 = min(self.x[1], other.x[1])

        y1 = max(self.y[0], other.y[0])
        y2 = min(self.y[1], other.y[1])

        # Handle the case where the two bounding boxes do not intersect
        if x2 < x1 or y2 < y1:
            return 0

        inter_area = (x2 - x1) * (y2 - y1)
        union_area = self.area + other.area - inter_area

        return inter_area / union_area

    @classmethod
    def from_center(cls: Type[Self], pt: Point, width: float, height: float) -> Self:
        """Create a bounding box from a center-point."""

        x_dist = width / 2
        x = (pt.x - x_dist, pt.x + x_dist)

        y_dist = height / 2
        y = (pt.y - y_dist, pt.y + y_dist)

        return BBox(x, y)


@dataclass(frozen=True)
class Object:
    bbox: BBox = field()
    history: list[BBox] = field(default_factory=list)
    ident: UUID = field(init=False, default_factory=uuid4)

    def best_match(self, bboxes: Iterable[BBox]) -> tuple[BBox, float]:
        scores = ((bbox, self.bbox.iou(bbox)) for bbox in bboxes)
        return max(scores, key=lambda s: s[1])

    def advance(self, bbox: BBox) -> Object:
        return Object(bbox, [self.bbox] + self.history)


class Trajectory:
    pass


def load_frames(capture: cv2.VideoCapture) -> Iterator[cv2.Mat]:
    if not capture.isOpened():
        raise RuntimeError("Capture is no longer opened")

    read_success, frame = capture.read()

    while read_success:
        yield frame
        read_success, frame = capture.read()

    capture.release()


def _bbox_from_tensor(tensor: torch.Tensor) -> BBox:
    """Create a bounding box from a torch tensor."""

    if tensor.ndim != 1:
        raise ValueError("Bounding box tensor must be 1-dimensional")

    if tensor.size(0) != 4:
        raise ValueError("Bounding box tensor must contain exactly 4 elements")

    x_interval = (tensor[0].item(), tensor[2].item())
    y_interval = (tensor[1].item(), tensor[3].item())

    return BBox(x_interval, y_interval)


def detect(model: models.MaskRCNN, image: torch.Tensor) -> set[BBox]:
    """Create a set of bounding boxes from objects in an image."""

    batch = image.unsqueeze(0)
    results = model(batch)
    result = results[0]

    return {_bbox_from_tensor(box) for box in result["boxes"]}


class Predictor(Protocol):
    def __call__(self, target: BBox) -> BBox:
        ...


@dataclass()
class PLine:
    """Parametric line."""

    delta_x: float = field()
    delta_y: float = field()
    bias: Point = field()

    def __call__(self, t: float) -> Point:
        return Point(self.delta_x * t + self.bias.x, self.delta_y * t + self.bias.y)

    def closest_point(self, point: Point) -> Point:
        """The point on the line closest to the given point."""

        return Point(0, 0)

    @classmethod
    def from_points(cls: Type[Self], p1: Point, p2: Point) -> Self:
        return cls(p2.x - p1.x, p2.y - p1.y, p1)


def _predict_linear(bbox: BBox, history: list[Point]) -> Predictor:
    line = PLine.from_points(bbox.center, history[0])

    def _predictor(target: BBox) -> BBox:
        pt = line.closest_point(target.center)
        return BBox.from_center(pt, bbox.width, bbox.height)

    return _predictor


def predict(obj: Object, method: str) -> Predictor:
    if len(obj.history) == 0:
        raise ValueError("Cannot predict the trajectory of an object with no history")

    if method == "linear":
        return _predict_linear(obj.bbox, [bbox.center for bbox in obj.history])

    raise ValueError(f"Unknown interpolation method {method}")


def track(objs: list[Object], bboxes: set[BBox], *, method: str, tol: float = 0.9) -> list[Object]:
    tracked = []

    # For every object assign the best detection match as its new current position
    for obj in objs:
        bbox, iou = obj.best_match(bboxes)

        if iou > tol:
            bboxes.remove(bbox)
            tracked.append(obj.advance(bbox))
        elif len(obj.history) > 0:
            predictor = predict(obj, method)
            closest_future_bbox = predictor(bbox)
            future_iou = closest_future_bbox.iou(bbox)

            if future_iou > tol:
                bboxes.remove(bbox)
                tracked.append(obj.advance(bbox))

    # Add the un-matched detections as new objects
    for bbox in bboxes:
        tracked.append(Object(bbox))

    return tracked


def show(frame: cv2.Mat, objects: Objects, trajectories: Trajectories):
    pass


def write(frame: cv2.Mat, objects: Objects, trajectories: Trajectories, video: cv2.VideoWriter):
    pass


@click.command()
@click.option("--no-show", is_flag=True, help="Do not show live video while running")
@click.option(
    "--record",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Record prediction data into a video file",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "pchip", "spline"]),
    default="linear",
    show_default=True,
    help="The interpolation method to use",
)
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def run(video: Path, interp: str, no_show: bool, record: Optional[Path]):
    weights = models.MaskRCNN_ResNet50_FPN_V2_Weights
    model = models.maskrcnn_resnet50_fpn_v2(weights=weights.DEFAULT)
    model.eval()

    capture = cv2.VideoCapture(str(video))
    frames = load_frames(capture)

    first_frame = next(frames)
    detections = detect(model, weights.transforms(first_frame))
    objects = [Object(box) for box in detections]

    if record:
        codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recording = cv2.VideoWriter(str(record), codec, fps, (width, height))
    else:
        recording = None

    for frame in frames:
        image = weights.transforms(frame)
        detections = detect(model, image)
        objects = track(objects, detections)
        trajectories = predict(objects, interp)

        if not no_show:
            show(frame, objects, trajectories)

        if recording:
            write(frame, objects, trajectories, recording)

    if recording:
        recording.close()


if __name__ == "__main__":
    run()
