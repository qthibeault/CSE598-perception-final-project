from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from signal import SIGINT, SIGTERM, signal
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Protocol, Type

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

    def distance(self, other: Point) -> float:
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))

    def as_tuple(self, *, dtype: Type[float] | Type[int] = float) -> tuple[float, float]:
        return (dtype(self.x), dtype(self.y))


@dataclass(frozen=True)
class BBox:
    """The bounding box of an object."""

    p1: Point = field()
    p2: Point = field()

    def __post_init__(self):
        if self.p1.x >= self.p2.x:
            raise ValueError("x0 must be strictly less than x1")

        if self.p1.y >= self.p2.y:
            raise ValueError("y0 must be strictly less than y1")

    @property
    def width(self) -> float:
        return self.p2.x - self.p1.x

    @property
    def height(self) -> float:
        return self.p2.y - self.p1.y

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point(self.p1.x + self.width / 2, self.p1.y + self.height / 2)

    def iou(self, other: BBox) -> float:
        """Compute the Intersection-over-Union (IOU) score between two bounding boxes."""

        x1 = max(self.p1.x, other.p1.x)
        x2 = min(self.p2.x, other.p2.x)

        y1 = max(self.p1.y, other.p1.y)
        y2 = min(self.p2.y, other.p2.y)

        # Handle the case where the two bounding boxes do not intersect
        if x2 < x1 or y2 < y1:
            return 0

        inter_area = (x2 - x1) * (y2 - y1)
        union_area = self.area + other.area - inter_area

        return inter_area / union_area

    def draw(self, img: cv2.Mat, color: tuple[int, int, int]):
        start = self.p1.as_tuple(dtype=int)
        end = self.p2.as_tuple(dtype=int)
        thickness = 2

        cv2.rectangle(img, start, end, color, thickness)

    @classmethod
    def from_center(cls: Type[Self], pt: Point, width: float, height: float) -> Self:
        """Create a bounding box from a center-point."""

        x_dist = width / 2
        y_dist = height / 2

        p1 = Point(pt.x - x_dist, pt.y - y_dist)
        p2 = Point(pt.x + x_dist, pt.y + y_dist)

        return BBox(p1, p2)


def _random_color() -> tuple[int, int, int]:
    return (randint(0, 255), randint(0, 255), randint(0, 255))


@dataclass(frozen=True)
class Object:
    bbox: BBox = field(hash=False)
    history: list[BBox] = field(default_factory=list, hash=False)
    color: tuple[int, int, int] = field(default_factory=_random_color)

    @property
    def id(self) -> int:
        return hash(self.color)

    @property
    def largest_bbox(self) -> BBox:
        if len(self.history) == 0:
            return self.bbox

        prev_best = max(self.history, key=lambda b: b.area)
        return max(self.bbox, prev_best, key=lambda b: b.area)

    def best_match(self, bboxes: Iterable[BBox]) -> tuple[BBox, float]:
        scores = ((bbox, self.bbox.iou(bbox)) for bbox in bboxes)
        return max(scores, key=lambda s: s[1])

    def advance(self, bbox: BBox) -> Object:
        return Object(bbox, [self.bbox] + self.history, self.color)

    def draw(self, img: cv2.Mat):
        self.bbox.draw(img, self.color)
        cv2.circle(
            img,
            center=self.bbox.center.as_tuple(dtype=int),
            radius=5,
            color=self.color,
            thickness=-1,
        )


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

    p1 = Point(tensor[0].item(), tensor[1].item())
    p2 = Point(tensor[2].item(), tensor[3].item())

    return BBox(p1, p2)


def detect(
    model: models.MaskRCNN,
    image: torch.Tensor,
    *,
    labels: list[str],
    allowed: list[str],
) -> set[BBox]:
    """Create a set of bounding boxes from objects in an image."""
    logger = logging.getLogger("detection")
    logger.debug("Beginning detection")

    batch = image.unsqueeze(0)
    results = model(batch)
    result = results[0]
    bbox_labels = [labels[int(label_idx.item())] for label_idx in result["labels"]]

    if len(allowed) == 0:
        detections = {_bbox_from_tensor(box) for box in result["boxes"]}
        logger.debug(f"Labels in frame: {bbox_labels}")
    else:
        detections = {
            _bbox_from_tensor(box)
            for box, label in zip(result["boxes"], bbox_labels)
            if label in allowed
        }

    logger.debug(f"Found {len(detections)} objects in frame")
    return detections


class Predictor(Protocol):
    def __call__(self, target: BBox) -> BBox:
        ...

    def best_match(self, bboxes: Iterable[BBox]) -> tuple[BBox, float]:
        ...

    def draw(self, img: cv2.Mat, color: tuple[int, int, int]) -> None:
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

        solutions: list[dict[str, float]] = solve(dot_prod, dict=True)
        assert len(solutions) == 1

        return self(solutions[0][t])

    @classmethod
    def from_points(cls: Type[Self], p1: Point, p2: Point) -> Self:
        return cls(p2.x - p1.x, p2.y - p1.y, p1)


@dataclass(frozen=True)
class LinearPredictor(Predictor):
    line: PLine
    bbox: BBox

    def __call__(self, target: BBox) -> BBox:
        pt = self.line.closest_point(target.center)
        return BBox.from_center(pt, self.bbox.width, self.bbox.height)

    def best_match(self, bboxes: Iterable[BBox]) -> tuple[BBox, float]:
        def _future_iou(bbox: BBox) -> float:
            future_bbox = self(bbox)
            return self.bbox.iou(future_bbox)

        scores = ((bbox, _future_iou(bbox)) for bbox in bboxes)
        return max(scores, key=lambda s: s[1])

    def draw(self, img: cv2.Mat, color: tuple[int, int, int]):
        p1 = self.line(0).as_tuple(dtype=int)
        p2 = self.line(1).as_tuple(dtype=int)
        p3 = self.line(100).as_tuple(dtype=int)

        cv2.circle(img, p1, radius=0, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, p2, radius=0, color=(0, 0, 255), thickness=-1)
        cv2.line(img, p1, p3, color=color, thickness=2)


def _predict_linear(obj: Object) -> Predictor:
    line = PLine.from_points(obj.history[0].center, obj.bbox.center)
    return LinearPredictor(line, obj.largest_bbox)


def predict(obj: Object, method: str) -> Predictor:
    if len(obj.history) == 0:
        raise ValueError("Cannot predict the trajectory of an object with no history")

    if method == "linear":
        return _predict_linear(obj)

    raise ValueError(f"Unknown interpolation method {method}")


@dataclass(frozen=True)
class TrackingResult:
    obj: Object = field()


@dataclass(frozen=True)
class ProxUpdate(TrackingResult):
    pass


@dataclass(frozen=True)
class PredUpdate(TrackingResult):
    predictor: Predictor = field(hash=False)


@dataclass(init=False, frozen=True)
class NewObject(TrackingResult):
    def __init__(self, bbox: BBox):
        super().__init__(Object(bbox))


@dataclass(frozen=True)
class Unchanged(TrackingResult):
    pass


def track(
    objs: set[Object],
    bboxes: set[BBox],
    *,
    method: str,
    tol: float = 0.8,
) -> list[TrackingResult]:
    logger = logging.getLogger("tracking")
    logger.debug(f"Beginning tracking with {len(objs)} objects and {len(bboxes)} detections")

    results: list[TrackingResult] = []
    unassigned_objs = set()

    for obj in objs:
        if len(bboxes) == 0:
            unassigned_objs.add(obj)
        else:
            bbox, iou = obj.best_match(bboxes)
            if iou > tol:
                logger.debug(f"Object {obj.id} assigned detection from IoU score")
                bboxes.remove(bbox)
                results.append(ProxUpdate(obj.advance(bbox)))
            else:
                unassigned_objs.add(obj)

    unassigned_objs2 = set()

    for obj in unassigned_objs:
        if len(bboxes) == 0:
            unassigned_objs2.add(obj)
        elif len(obj.history) > 0:
            predictor = predict(obj, method)
            bbox, future_iou = predictor.best_match(bboxes)

            if future_iou > tol:
                logger.debug(f"Object {obj.id} assigned detection from trajectory prediction")
                bboxes.remove(bbox)
                results.append(PredUpdate(obj.advance(bbox), predictor))
            else:
                unassigned_objs2.add(obj)
        else:
            unassigned_objs2.add(obj)

    logger.debug(f"{len(unassigned_objs2)} objects not updated")

    for obj in unassigned_objs:
        results.append(Unchanged(obj))

    logger.debug(f"{len(bboxes)} detections not assigned to any object")

    for bbox in bboxes:
        results.append(NewObject(bbox))

    return results


def draw(frame: cv2.Mat, results: Iterable[TrackingResult]):
    for result in results:
        obj = result.obj

        if isinstance(result, (ProxUpdate, PredUpdate)):
            obj.draw(frame)

        if isinstance(result, PredUpdate):
            result.predictor.draw(frame, obj.color)

        if isinstance(result, Unchanged) and len(obj.history) > 2:
            gray = (220, 220, 220)
            line = PLine.from_points(obj.history[0].center, obj.bbox.center)
            predictor = LinearPredictor(line, obj.bbox)

            obj.bbox.draw(frame, color=gray)
            predictor.draw(frame, color=gray)


def _handle_shutdown(video: cv2.VideoCapture, recording: Optional[cv2.VideoWriter]):
    def handler(*args: Any):
        logger = logging.getLogger("exit")
        logger.debug("SIGINT detected, stopping processing")

        if video.isOpened():
            video.release()

        if recording and recording.isOpened():
            recording.release()

        cv2.destroyAllWindows()
        exit(0)

    return handler


@click.command()
@click.option("--show/--no-show", default=True, help="Do not show live video while running")
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
@click.option("--debug", is_flag=True, help="Show debugging output during execution")
@click.option("--labels", multiple=True, help="List of detection labels to accept")
@click.option("--captures", multiple=True, type=int, help="List of frames to capture into images")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def run(
    video: Path,
    method: str,
    show: bool,
    record: Optional[Path],
    debug: bool,
    labels: list[str],
    captures: list[int],
):
    logger = logging.getLogger("main")
    weights = models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = models.maskrcnn_resnet50_fpn_v2(weights=weights)
    transforms = weights.transforms()
    categories = weights.meta["categories"]
    model.eval()

    logging.basicConfig(level=logging.DEBUG if debug else logging.WARN)
    logger.debug("Starting frame 0 processing")

    capture = cv2.VideoCapture(str(video))

    if record:
        codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recording = cv2.VideoWriter(str(record), codec, fps, (width, height))
    else:
        recording = None

    signal(SIGINT, _handle_shutdown(capture, recording))
    signal(SIGTERM, _handle_shutdown(capture, recording))

    frames = load_frames(capture)
    first_frame = next(frames)
    transformed = transforms(Image.fromarray(first_frame))
    detections = detect(model, transformed, labels=categories, allowed=labels)
    track_results = [NewObject(box) for box in detections]

    draw(first_frame, track_results)

    if show:
        cv2.imshow("video", first_frame)
        cv2.waitKey(1)

    if recording:
        recording.write(first_frame)

    for idx, frame in enumerate(frames, start=1):
        logger.debug(f"Starting frame {idx} processing")

        transformed = transforms(Image.fromarray(frame))
        detections = detect(model, transformed, labels=categories, allowed=labels)

        if len(detections) > 0:
            objects = {result.obj for result in track_results}
            track_results = track(objects, detections, method=method)
            draw(frame, track_results)

        if show:
            cv2.imshow("video", frame)
            cv2.waitKey(1)

        if recording:
            recording.write(frame)

        if idx in captures:
            cv2.imwrite(f"{video.stem}_frame{idx}.png", frame)

    if recording:
        recording.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
