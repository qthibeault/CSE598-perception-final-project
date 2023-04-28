from __future__ import annotations

import logging
from collections.abc import Iterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from signal import SIGINT, SIGTERM, signal
from typing import TYPE_CHECKING, Any, Optional

import click
import cv2
import torchvision.models.detection as models
from PIL import Image
from scipy import stats

if TYPE_CHECKING:
    import torch

from detection import BBox, Color, Detection, Point, Tracker
from prediction import Predictor, LinearPredictor, LinearRegressionPredictor, NonlinearPredictor


@dataclass(frozen=True, slots=True)
class RawFrame:
    image: cv2.Mat
    index: int


def load_frames(capture: cv2.VideoCapture) -> Iterator[RawFrame]:
    if not capture.isOpened():
        raise RuntimeError("Capture is no longer opened")

    index = 0
    read_success, image = capture.read()

    while read_success:
        yield RawFrame(image, index)
        index += 1
        read_success, image = capture.read()

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
    model: models.MaskRCNN, img: torch.Tensor, *, labels: list[str], allowed: list[str]
) -> Iterable[BBox]:
    """Create a set of bounding boxes from objects in an image."""

    if len(allowed) > len(labels):
        raise ValueError("Cannot allow more labels than provided")

    logger = logging.getLogger("detection")
    logger.debug("Beginning detection")

    batch = img.unsqueeze(0)
    results = model(batch)
    result = results[0]
    bbox_labels = [labels[int(label_idx.item())] for label_idx in result["labels"]]

    if len(allowed) == 0:
        logger.debug(f"Labels in frame: {bbox_labels}")

    for coords, obj_class in zip(result["boxes"], result["labels"]):
        bbox = _bbox_from_tensor(coords)
        obj_class = int(obj_class.item())
        label = labels[obj_class]

        if len(allowed) > 0 and label not in allowed:
            continue

        yield bbox
        logger.debug(f"Detected {label}")


def predict(tracker: Tracker, method: str) -> Predictor:
    if len(tracker.history) == 0:
        raise ValueError("Cannot predict the trajectory of an object with no history")

    if method == "linear":
        return LinearPredictor(tracker.containing_bbox, tracker.position, tracker.history[0])
    elif method == "linreg":
        return LinearRegressionPredictor(tracker.containing_bbox, tracker.detections)
    elif method == "nonlinear":
        return NonlinearPredictor(tracker.containing_bbox, tracker.detections)

    raise ValueError(f"Unknown interpolation method {method}")


@dataclass(frozen=True)
class TrackingResult:
    tracker: Tracker = field()


@dataclass(frozen=True)
class ProxUpdate(TrackingResult):
    pass


@dataclass(frozen=True)
class PredUpdate(TrackingResult):
    predictor: Predictor = field(hash=False)


@dataclass(init=False, frozen=True)
class NewObject(TrackingResult):
    def __init__(self, bbox: BBox, frame: int):
        super().__init__(Tracker(Detection(bbox, frame)))


@dataclass(frozen=True)
class Unchanged(TrackingResult):
    pass


def track(
    prev: list[Tracker], bboxes: Iterable[BBox], frame: int, *, method: str, tol: float = 0.8
) -> list[TrackingResult]:
    bboxes = set(bboxes)
    logger = logging.getLogger("tracking")
    logger.debug(f"Beginning tracking with {len(prev)} objects and {len(bboxes)} detections")

    results: list[TrackingResult] = []
    unassigned = set()

    # Try to associate detected bounding boxed via proximity
    for tracker in prev:
        if len(bboxes) == 0:
            unassigned.add(tracker)
        else:
            bbox, iou = max(tracker.score_bboxes(bboxes), key=lambda s: s[1])

            if iou > tol:
                det = Detection(bbox, frame)
                updated = tracker.step(det)
                results.append(ProxUpdate(updated))
                bboxes.remove(bbox)
                logger.debug(f"Object {hash(tracker)} assigned detection from IoU score")
            else:
                unassigned.add(tracker)

    unassigned2 = set()

    # Try to associate detected bounding boxed via future proximity
    for tracker in unassigned:
        if len(bboxes) == 0 or len(tracker.history) == 0:
            unassigned2.add(tracker)
        else:
            predictor = predict(tracker, method)
            bbox, future_iou = predictor.bbox_scores(frame, bboxes)

            if future_iou > tol:
                det = Detection(bbox, frame)
                updated = tracker.step(det)
                logger.debug(
                    f"Object {hash(tracker)} assigned detection from trajectory prediction"
                )
                bboxes.remove(bbox)
                results.append(PredUpdate(updated, predictor))
            else:
                unassigned2.add(tracker)

    logger.debug(f"{len(unassigned2)} objects not updated")

    for obj in unassigned2:
        results.append(Unchanged(obj))

    logger.debug(f"{len(bboxes)} detections not assigned to any object")

    for bbox in bboxes:
        results.append(NewObject(bbox, frame))

    return results


def draw(frame: cv2.Mat, method: str, results: Iterable[TrackingResult]):
    for result in results:
        tracker = result.tracker

        if isinstance(result, (ProxUpdate, PredUpdate)):
            tracker.draw(frame)

        if isinstance(result, Unchanged) and len(tracker.history) > 5:
            gray = Color(220, 220, 220)
            predictor = predict(tracker, method)

            tracker.position.draw(frame, color=gray)
            predictor.draw(frame, color=gray, from_frame=tracker.position.frame)


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
    type=click.Choice(["linear", "linreg", "nonlinear"]),
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
        codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
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
    transformed = transforms(Image.fromarray(first_frame.image))
    detections = detect(model, transformed, labels=categories, allowed=labels)
    results = [NewObject(box, first_frame.index) for box in detections]

    draw(first_frame.image, method, results)

    if show:
        cv2.imshow("video", first_frame.image)
        cv2.waitKey(1)

    if recording:
        recording.write(first_frame)

    for frame in frames:
        logger.debug(f"Starting frame {frame.index} processing")

        transformed = transforms(Image.fromarray(frame.image))
        detections = detect(model, transformed, labels=categories, allowed=labels)
        detections = list(detections)

        if len(detections) > 0:
            trackers = [result.tracker for result in results]
            results = track(trackers, detections, frame.index, method=method)
            draw(frame.image, method, results)

        if show:
            cv2.imshow("video", frame.image)
            cv2.waitKey(1)

        if recording:
            recording.write(frame.image)

        if frame.index in captures:
            cv2.imwrite(f"{video.stem}_frame{frame.index}.png", frame.image)

    if recording:
        recording.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
