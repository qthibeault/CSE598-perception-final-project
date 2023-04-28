from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from math import sqrt
from random import randint
from typing import Type

import cv2
from numpy import float64, ndarray
from numpy.typing import NDArray
from typing_extensions import Self


@dataclass(frozen=True, slots=True)
class Color:
    R: int = field()
    G: int = field()
    B: int = field()

    def as_tuple(self) -> tuple[int, int, int]:
        return self.R, self.G, self.B

    def __hash__(self) -> int:
        return hash((self.R, self.G, self.B))


@dataclass(frozen=True, slots=True)
class Point(Iterable[float]):
    x: float
    y: float

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y))

    def distance(self, other: Point) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def as_tuple(self, *, dtype: Type[float] | Type[int] = float) -> tuple[float, float]:
        return (dtype(self.x), dtype(self.y))

    def draw(self, img: cv2.Mat, color: Color):
        cv2.circle(
            img,
            center=self.as_tuple(),
            radius=5,
            color=color.as_tuple(),
            thickness=-1,
        )

    @classmethod
    def from_ndarray(cls, arr: NDArray[float64]) -> Self:
        if not isinstance(arr, ndarray):
            raise TypeError("arr must be ndarray")

        if arr.shape != (2,):
            raise ValueError("arr must have shape (2,)")

        return cls(float(arr[0]), float(arr[1]))


@dataclass(frozen=True, slots=True)
class BBox(Iterable[Point]):
    """The bounding box of an object."""

    p1: Point = field()
    p2: Point = field()

    def __post_init__(self):
        if self.p1.x >= self.p2.x:
            raise ValueError("x0 must be strictly less than x1")

        if self.p1.y >= self.p2.y:
            raise ValueError("y0 must be strictly less than y1")

    def __hash__(self) -> int:
        return hash((self.p1, self.p2))

    def __iter__(self) -> Iterator[Point]:
        return iter((self.p1, self.p2))

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

    def recenter(self, new_center: Point) -> Self:
        x_dist = self.width / 2
        y_dist = self.height / 2

        p1 = Point(new_center.x - x_dist, new_center.y - y_dist)
        p2 = Point(new_center.x + x_dist, new_center.y + y_dist)

        return BBox(p1, p2)

    def draw(self, img: cv2.Mat, color: Color):
        start = self.p1.as_tuple(dtype=int)
        end = self.p2.as_tuple(dtype=int)
        thickness = 2

        cv2.rectangle(img, start, end, color, thickness)


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    frame: int

    def __hash__(self) -> int:
        return hash((self.bbox, self.frame))

    def draw(self, img: cv2.Mat, color: Color):
        self.bbox.draw(img, color)


def _random_color() -> Color:
    return Color(randint(0, 255), randint(0, 255), randint(0, 255))


@dataclass(frozen=True, slots=True)
class Tracker:
    position: Detection = field()
    history: list[Detection] = field(default_factory=list)
    color: Color = field(default_factory=_random_color)

    def __hash__(self) -> int:
        return hash(self.color)

    @property
    def bboxes(self) -> Iterable[BBox]:
        yield self.position.bbox
        yield from [d.bbox for d in self.history]

    @property
    def containing_bbox(self) -> BBox:
        if len(self.history) == 0:
            return self.position.bbox

        prev_best = max((d.bbox for d in self.history), key=lambda b: b.area)
        return max(self.position.bbox, prev_best, key=lambda b: b.area)

    @property
    def last_frame(self) -> int:
        return self.position.frame

    def step(self, d: Detection) -> Tracker:
        return Tracker(d, [self.position] + self.history, self.color)

    def draw(self, img: cv2.Mat):
        self.position.bbox.draw(img, self.color)
        self.position.bbox.center.draw(img, self.color)
