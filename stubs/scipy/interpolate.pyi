from typing import Literal, Optional, TYPE_CHECKING

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

class interp1d:
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        kind: str | int = ...,
        axis: int = ...,
        copy: bool = ...,
        bounds_error: bool = ...,
        fill_value: ArrayLike | tuple[ArrayLike, ArrayLike] | Literal["extrapolate"] = ...,
        assume_sorted: bool = ...,
    ): ...
    def __call__(self, x: ArrayLike) -> NDArray[float64]: ...

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

class RectBivariateSpline:
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        bbox: ArrayLike | None = ...,
        kx: int | None = ...,
        ky: int | None = ...,
        s: float | None = ...,
    ): ...
    def __call__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        dx: int = ...,
        dy: int = ...,
        grid: bool = ...,
    ) -> NDArray[float64]: ...

if TYPE_CHECKING:
    DerivativeOrder = tuple[float, float]
    BCAlias = Literal["clamped", "natural", "not-a-knot", "periodic"]
    BCType = tuple[DerivativeOrder | BCAlias, DerivativeOrder | BCAlias]

def make_interp_spline(
    x: ArrayLike,
    y: ArrayLike,
    k: int = ...,
    t: ArrayLike = ...,
    bc_type: Optional[BCType] = ...,
    axis: int = ...,
    check_finite: bool = ...,
) -> BSpline: ...
