from pathlib import Path
from typing import Iterator, Optional

import click
import cv2
import torchvision.models.detection as models


class Detections:
    pass


class Objects:
    pass


class Trajectories:
    pass


def load_frames(capture: cv2.VideoCapture) -> Iterator[cv2.Mat]:
    read_success, frame = capture.read()

    while read_success:
        yield frame
        read_success, frame = capture.read()

    capture.release()


def detect(model: models.MaskRCNN, frame: cv2.Mat) -> Detections:
    pass


def track(prev: Detections, curr: Detections) -> Objects:
    pass


def predict(objects: Objects, method: str) -> Trajectories:
    pass


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
@click.argument(
    "video",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def run(video: Path, interp: str, no_show: bool, record: Optional[Path]):
    weights = models.MaskRCNN_ResNet50_FPN_V2_Weights
    model = models.maskrcnn_resnet50_fpn_v2(weights=weights.DEFAULT)
    capture = cv2.VideoCapture(str(video))
    frames = load_frames(capture)
    prev_detections = detect(model, next(frames))

    if record:
        codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recording = cv2.VideoWriter(str(record), codec, fps, (width, height))
    else:
        recording = None

    for frame in frames:
        curr_detections = detect(model, frame)
        objects = track(prev_detections, curr_detections)
        trajectories = predict(objects, interp)
        prev_detections = curr_detections

        if not no_show:
            show(frame, objects, trajectories)

        if recording:
            write(frame, objects, trajectories, recording)


if __name__ == "__main__":
    run()
