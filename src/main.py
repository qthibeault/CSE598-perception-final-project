from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Protocol, Type
from uuid import UUID, uuid4

import click
import cv2
import torchvision.models.detection as models
from PIL import Image
from sympy import solve, symbols

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


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

    def draw(self, img: cv2.Mat):
        start = (self.x[0], self.y[0])
        end = (self.x[1], self.y[1])
        color = (255, 0, 0)
        thickness = 2

        cv2.rectangle(img, start, end, color, thickness)

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

    def draw(self, img: cv2.Mat):
        self.bbox.draw(img)


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

    def draw(self, img: cv2.Mat) -> None:
        ...


@dataclass(frozen=True)
class PLine:
    """Parametric line."""

    delta_x: float = field()
    delta_y: float = field()
    bias: Point = field()

    def __call__(self, t: float) -> Point:
        return Point(self.delta_x * t + self.bias.x, self.delta_y * t + self.bias.y)

    def closest_point(self, point: Point) -> Point:
        """The point on the line closest to the given point.

        We can compute any point P on a parametric line L using the formula
        `(x0, y0) + (x2 - x1, y2 - y1) * t`. We can define the line from any point P on line L to
        the target point Q as `P - Q`. The line from the closest point P* on line L to the target
        point Q will be perpendicular. Therefore, the dot product between the line `P - Q` and the
        line L will be 0. Given this, we can formulate the equation for the closest point P* in
        terms of parametric variable `t` as
        `(x0 + (x2 - x1) * t - Rx) * (x2 - x1) + (y0 + (y2 - y1) * t - Ry) * (y2 - y1) = 0`.
        """

        t = symbols("t")
        x_eq = self.delta_x * t + self.bias.x
        y_eq = self.delta_y * t + self.bias.y

        x_perp_eq = x_eq - point.x
        y_perp_eq = y_eq - point.y
        dot_prod = x_perp_eq * self.delta_x + y_perp_eq * self.delta_y

        sol = solve(dot_prod, dict=True)

        return self(sol["t"])

    @classmethod
    def from_points(cls: Type[Self], p1: Point, p2: Point) -> Self:
        return cls(p2.x - p1.x, p2.y - p1.y, p1)


@dataclass(frozen=True)
class LinearPredictor(Predictor):
    line: PLine
    width: float
    height: float

    def __call__(self, target: BBox) -> BBox:
        pt = self.line.closest_point(target.center)
        return BBox.from_center(pt, self.width, self.height)

    def draw(self, img: cv2.Mat):
        p1 = self.line(0).as_tuple()
        p2 = self.line(1).as_tuple()
        p3 = self.line(100).as_tuple()

        cv2.circle(img, p1, radius=0, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, p2, radius=0, color=(0, 0, 255), thickness=-1)
        cv2.line(img, p1, p3, color=(0, 255, 0), thickness=2)


def _predict_linear(curr_bbox: BBox, prev_centers: list[Point]) -> Predictor:
    line = PLine.from_points(prev_centers[0], curr_bbox.center)
    return LinearPredictor(line, curr_bbox.width, curr_bbox.height)


def predict(obj: Object, method: str) -> Predictor:
    if len(obj.history) == 0:
        raise ValueError("Cannot predict the trajectory of an object with no history")

    if method == "linear":
        return _predict_linear(obj.bbox, [bbox.center for bbox in obj.history])

    raise ValueError(f"Unknown interpolation method {method}")


def track(
    objs: Iterable[Object],
    bboxes: set[BBox],
    *,
    method: str,
    tol: float = 0.9,
) -> tuple[list[Object], list[Predictor]]:
    tracked = []
    predictors = []

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
                predictors.append(predictor)

    # Add the un-matched detections as new objects
    for bbox in bboxes:
        tracked.append(Object(bbox))

    return tracked, predictors


def show(
    frame: cv2.Mat,
    objects: Iterable[Object],
    predictors: Iterable[Predictor],
    *,
    win_title: str = "Video",
):
    for obj in objects:
        print(obj)
        obj.draw(frame)

    for predictor in predictors:
        predictor.draw(frame)

    cv2.imshow(win_title, frame)


def write(
    rec: cv2.VideoWriter,
    frame: cv2.Mat,
    objects: Iterable[Object],
    predictors: Iterable[Predictor],
    *,
    should_draw: bool = False
):
    if should_draw:
        for obj in objects:
            obj.draw(frame)

        for predictor in predictors:
            predictor.draw(frame)

    rec.write(frame)


@click.command()
@click.option("--no-show", is_flag=True, help="Do not show live video while running")
@click.option(
    "--record",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Record prediction data into a video file",
)
@click.option(
    "--method",
    type=click.Choice(["linear", "pchip", "spline"]),
    default="linear",
    show_default=True,
    help="The interpolation method to use",
)
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def run(video: Path, method: str, no_show: bool, record: Optional[Path]):
    weights = models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = models.maskrcnn_resnet50_fpn_v2(weights=weights)
    transforms = weights.transforms()
    model.eval()

    capture = cv2.VideoCapture(str(video))
    frames = load_frames(capture)
    first_frame = next(frames)
    transformed = transforms(Image.fromarray(first_frame))
    detections = detect(model, transformed)
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
        transformed = transforms(Image.fromarray(frame))
        detections = detect(model, transformed)
        objects, predictors = track(objects, detections, method=method)

        if not no_show:
            show(frame, objects, predictors)

        if recording:
            write(recording, frame, objects, predictors, should_draw=no_show)

    if recording:
        recording.release()

    if not no_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
