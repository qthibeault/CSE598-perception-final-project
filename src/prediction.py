from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, optimize, stats
from typing_extensions import Self

from detection import BBox, Color, Detection, Point


class Predictor(Protocol):
    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> tuple[BBox, float]:
        ...

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1) -> None:
        ...


class LinearPredictor(Predictor):
    def __init__(self, bbox: BBox, d1: Detection, d2: Detection):
        self.bbox = bbox
        self.d1 = d1
        self.d2 = d2
        self.interp = interpolate.interp1d(
            x=np.array([d1.frame, d2.frame]),
            y=np.array([d1.bbox.center.as_tuple(), d2.bbox.center.as_tuple()]),
            kind="linear",
            fill_value="extrapolate",
        )

    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> Iterable[tuple[BBox, float]]:
        center = Point.from_ndarray(self.interp(frame))
        future_bbox = self.bbox.recenter(center)

        for bbox in bboxes:
            yield bbox, future_bbox.iou(bbox)

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1):
        p1 = self.interp(self.d1.frame)
        p1 = (float(p1[0]), float(p1[1]))

        p2 = self.interp(self.d2.frame)
        p2 = (float(p2[0]), float(p2[1]))

        p3 = self.interp(self.d2.frame + 100)
        p3 = (float(p3[0]), float(p3[1]))

        cv2.circle(img, p1, radius=0, color=color, thickness=-1)
        cv2.circle(img, p2, radius=0, color=color, thickness=-1)
        cv2.line(img, p1, p3, color=color, thickness=2)


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

        cv2.line(img, p1, p2, color=color, thickness=2)


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

        n_pts = 20 + len(self.history) * 5
        frame_pts = np.linspace(from_frame, to_frame, n_pts)
        line_pts = [self.interp(pt) for pt in frame_pts]
        line_pts = [(int(pt[0]), int(pt[1])) for pt in line_pts]

        cv2.polylines(img, line_pts, isClosed=False, color=color, thickness=2)
