from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable

import cv2
import numpy as np
from scipy import interpolate, stats

from detection import BBox, Color, Detection, Point


def _interp_pt(
    interp: interpolate.interp1d, frame: int, *, height: float, width: float
) -> tuple[int, int]:
    pt = interp(frame)
    x = min(pt[0], width)
    y = min(pt[1], height)
    return (int(x), int(y))


class Predictor(Protocol):
    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> Iterable[tuple[BBox, float]]:
        ...

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1) -> None:
        ...


@dataclass()
class PLine:
    slope: tuple[float, float]
    point: Point

    def __call__(self, t: float) -> Point:
        return Point(self.slope[0] * t, self.slope[1] * t) + self.point


class LinearPredictor(Predictor):
    def __init__(self, bbox: BBox, d1: Detection, d2: Detection):
        upper = max(d1, d2, key=lambda d: d.frame)
        lower = min(d1, d2, key=lambda d: d.frame)
        slope = upper.bbox.center - lower.bbox.center

        self.bbox = bbox
        self.d1 = lower
        self.d2 = upper
        self.line = PLine(slope.as_tuple(), lower.bbox.center)

    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> Iterable[tuple[BBox, float]]:
        center = self.line(frame)
        future_bbox = self.bbox.recenter(center)

        for bbox in bboxes:
            yield bbox, future_bbox.iou(bbox)

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1):
        c = color.as_tuple()
        to_frame = 25

        p1 = self.d1.bbox.center.as_tuple(dtype=int)
        p2 = self.d2.bbox.center.as_tuple(dtype=int)
        p3 = self.line(to_frame).as_tuple(dtype=int)

        cv2.circle(img, p1, radius=4, color=c, thickness=-1)
        cv2.circle(img, p2, radius=4, color=c, thickness=-1)
        cv2.line(img, p1, p3, color=c, thickness=2)


class LinearRegressionPredictor(Predictor):
    def __init__(self, bbox: BBox, history: Iterable[Detection]):
        self.bbox = bbox
        self.history = list(history)
        self.x_interp = stats.linregress([(d.frame, d.bbox.center.x) for d in self.history])
        self.y_interp = stats.linregress([(d.frame, d.bbox.center.y) for d in self.history])

    def frame_center(self, frame: int) -> Point:
        x = self.x_interp.slope * frame + self.x_interp.intercept
        y = self.y_interp.slope * frame + self.y_interp.intercept

        return Point(x, y)

    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> Iterable[tuple[BBox, float]]:
        center = self.frame_center(frame)
        future_bbox = self.bbox.recenter(center)

        for bbox in bboxes:
            yield bbox, future_bbox.iou(bbox)

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1):
        last_frame = max(d.frame for d in self.history)
        to_frame = last_frame + 100

        if from_frame > last_frame:
            raise ValueError(f"from_frame cannot be greater than the last frame {last_frame}")

        if from_frame < 0:
            from_frame = min(d.frame for d in self.history)

        p1 = self.frame_center(from_frame).as_tuple()
        p2 = self.frame_center(to_frame).as_tuple()

        cv2.line(img, p1, p2, color=color.as_tuple(), thickness=2)


class NonlinearPredictor(Predictor):
    def __init__(self, bbox: BBox, history: Iterable[Detection]):
        self.bbox = bbox
        self.history = list(history)
        self.interp = interpolate.interp1d(
            x=np.array([d.frame for d in history]),
            y=np.array([d.bbox.center.as_tuple() for d in history]),
            kind="cubic",
            fill_value="extrapolate",
        )

    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> Iterable[tuple[BBox, float]]:
        center = Point.from_ndarray(self.interp(frame))
        future_bbox = self.bbox.recenter(center)

        for bbox in bboxes:
            yield bbox, future_bbox.iou(bbox)

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1):
        last_frame = max(d.frame for d in self.history)
        to_frame = last_frame + 100

        if from_frame > last_frame:
            raise ValueError(f"from_frame cannot be greater than the last frame {last_frame}")

        if from_frame < 0:
            from_frame = min(d.frame for d in self.history)

        h, w, _ = img.shape
        n_pts = 20 + len(self.history) * 5
        frame_pts = np.linspace(from_frame, to_frame, n_pts)
        line_pts = [_interp_pt(self.interp, pt, height=h, width=w) for pt in frame_pts]

        cv2.polylines(img, line_pts, isClosed=False, color=color.as_tuple(), thickness=2)
