from typing import Any, Callable, Literal

from numpy import float64
from numpy.typing import NDArray

class OptimizeResult:
    @property
    def x(self) -> NDArray[float64]: ...
    @property
    def success(self) -> bool: ...
    @property
    def status(self) -> int: ...
    @property
    def message(self) -> str: ...
    @property
    def fun(self) -> NDArray[float64]: ...
    @property
    def jac(self) -> NDArray[float64]: ...
    @property
    def hess(self) -> NDArray[float64]: ...
    @property
    def nfev(self) -> int: ...
    @property
    def njev(self) -> int: ...
    @property
    def nhev(self) -> int: ...
    @property
    def nit(self) -> int: ...
    @property
    def maxcv(self) -> float: ...

def minimize(
    fun: Callable[[NDArray[float64]], float],
    x0: NDArray[float64],
    args: tuple[Any, ...] | None = ...,
    method: None
    | Literal[
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ] = ...,
) -> OptimizeResult: ...
