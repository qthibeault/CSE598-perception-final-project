from typing import Literal

from numpy.typing import ArrayLike

class LinregressResult:
    @property
    def slope(self) -> float: ...
    @property
    def intercept(self) -> float: ...

def linregress(
    x: ArrayLike,
    y: ArrayLike | None = ...,
    alternative: Literal["two-sided", "less", "greater"] | None = ...,
) -> LinregressResult: ...
