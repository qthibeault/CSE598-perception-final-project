from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable

import cv2
import numpy as np
from scipy import interpolate, stats

from detection import BBox, Color, Detection, Point


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

        self.bbox = bbox
        self.d1 = lower
        self.d2 = upper
        self.spline = interpolate.make_interp_spline(
            x=[lower.frame, upper.frame],
            y=[lower.bbox.center.as_tuple(), upper.bbox.center.as_tuple()],
            k=1,
            # bc_type=("clamped", "clamped"),
        )

    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> Iterable[tuple[BBox, float]]:
        pt = self.spline(frame)
        center = Point.from_ndarray(pt)
        future_bbox = self.bbox.recenter(center)

        for bbox in bboxes:
            yield bbox, future_bbox.iou(bbox)

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1):
        c = color.as_tuple()
        to_frame = self.d2.frame + 25

        p1 = self.d1.bbox.center.as_tuple(dtype=int)
        p2 = self.d2.bbox.center.as_tuple(dtype=int)
        p3 = self.spline(to_frame).astype(int)
        p3 = (p3[0], p3[1])

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

        p1 = self.frame_center(from_frame).as_tuple(dtype=int)
        p2 = self.frame_center(to_frame).as_tuple(dtype=int)

        cv2.line(img, p1, p2, color=color.as_tuple(), thickness=2)


class NonlinearPredictor(Predictor):
    def __init__(self, bbox: BBox, history: Iterable[Detection]):
        self.bbox = bbox
        self.history = sorted(history, key=lambda d: d.frame)

        if len(self.history) == 2:
            k = 1
        elif len(self.history) == 3:
            k = 2
        elif len(self.history) > 3:
            k = 3
        else:
            raise ValueError("history must have >= 2 elements")

        self.x_spline = interpolate.make_interp_spline(
            x=[d.frame for d in self.history],
            y=[d.bbox.center.x for d in self.history],
            k=k,
        )
        self.y_spline = interpolate.make_interp_spline(
            x=[d.frame for d in self.history],
            y=[d.bbox.center.y for d in self.history],
            k=k,
        )

    def bbox_scores(self, frame: int, bboxes: Iterable[BBox]) -> Iterable[tuple[BBox, float]]:
        pt_x = self.x_spline(frame)
        pt_y = self.y_spline(frame)
        center = Point(float(pt_x), float(pt_y))
        future_bbox = self.bbox.recenter(center)

        for bbox in bboxes:
            yield bbox, future_bbox.iou(bbox)

    def draw(self, img: cv2.Mat, *, color: Color, from_frame: int = -1):
        last_frame = self.history[-1].frame
        to_frame = last_frame + 10

        if from_frame > last_frame:
            raise ValueError(f"from_frame cannot be greater than the last frame {last_frame}")

        if from_frame < 0:
            from_frame = self.history[0].frame

        h, w, _ = img.shape
        n_pts = 30

        for d in self.history:
            d.bbox.center.draw(img, color=color)

        frame_pts = np.linspace(from_frame, to_frame, n_pts)
        spline_pts = ((self.x_spline(pt), self.y_spline(pt)) for pt in frame_pts)
        line_pts = np.array([(int(pt[0]), int(pt[1])) for pt in spline_pts])
        line_pts = line_pts.clip((0, 0), (w, h))

        cv2.polylines(img, [line_pts], isClosed=False, color=color.as_tuple(), thickness=2)
