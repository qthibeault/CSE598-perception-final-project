from typing import Literal

from numpy import float64
from numpy.typing import ArrayLike, NDArray

def splprep(
    x: list[tuple[float, float]],
    w: ArrayLike | None = ...,
    ub: int | None = ...,
    ue: int | None = ...,
    k: int | None = ...,
    task: int | None = ...,
    s: float | None = ...,
    t: int | None = ...,
    full_output: int | None = ...,
    nest: int | None = ...,
    per: int | None = ...,
    quiet: int | None = ...,
) -> tuple[
    tuple[NDArray[float64], list[NDArray[float64]], int],
    NDArray[float64],
    float,
    int,
    str,
]: ...

class BSpline:
    def __init__(
        self,
        t: NDArray[float64],
        c: list[NDArray[float64]],
        k: int,
        extrapolate: bool | Literal["periodic"] | None = ...,
        axis: int | None = ...,
    ): ...
    def __call__(
        self,
        x: ArrayLike,
        nu: int | None = ...,
        extrapolate: bool | Literal["periodic"] | None = ...,
    ) -> NDArray[float64]: ...
    def derivative(self, nu: int | None = ...) -> BSpline: ...
