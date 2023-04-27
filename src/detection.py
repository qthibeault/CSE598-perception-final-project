from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from random import randint
from typing import Iterable, Type

import cv2
from numpy import float64, ndarray
from numpy.typing import NDArray
from typing_extensions import Self


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance(self, other: Point) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def as_tuple(self, *, dtype: Type[float] | Type[int] = float) -> tuple[float, float]:
        return (dtype(self.x), dtype(self.y))

    @classmethod
    def from_ndarray(cls, arr: NDArray[float64]) -> Self:
        if not isinstance(arr, ndarray):
            raise TypeError("arr must be ndarray")

        if arr.shape != (2,):
            raise ValueError("arr must have shape (2,)")

        return cls(float(arr[0]), float(arr[1]))


@dataclass(frozen=True)
class BBox:
    """The bounding box of an object."""

    p1: Point = field()
    p2: Point = field()
    n_frame: int = field()

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
    def from_center(cls: Type[Self], pt: Point, width: float, height: float, n_frame: int) -> Self:
        """Create a bounding box from a center-point."""

        x_dist = width / 2
        y_dist = height / 2

        p1 = Point(pt.x - x_dist, pt.y - y_dist)
        p2 = Point(pt.x + x_dist, pt.y + y_dist)

        return BBox(p1, p2, n_frame)


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

    @property
    def bboxes(self) -> Iterable[BBox]:
        yield self.bbox
        yield from self.history

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
