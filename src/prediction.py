from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Iterable, Type

import cv2
from sympy import symbols, solve
from typing_extensions import Self

from detection import BBox, Object, Point


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

    @classmethod
    def from_obj(cls, obj: Object) -> Self:
        line = PLine.from_points(obj.history[0].center, obj.bbox.center)
        return cls(line, obj.largest_bbox)
