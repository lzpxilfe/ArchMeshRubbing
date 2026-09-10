"""scipy.ndimage 두 함수의 좁은 타입 선언을 한자리에서 메운다.

Two of the ndimage calls this repository leans on are typed more narrowly in
scipy's stubs than the functions themselves are, and the mismatch is not a
mistake in the calling code:

* ``label`` hands back ``(labels, count)`` whenever no output array is given,
  but its declared return type also admits a bare ``int``, so every unpack
  reads as an error.
* ``gaussian_filter`` takes ``order`` per axis as a sequence - that is the
  only way to ask for a directional derivative, and this repository asks for
  them by the dozen when it reads a stroke's ridge - while the stub declares
  a single ``int``.

Rather than scatter a suppression at each of a dozen call sites, the two gaps
are named once here, with the reason, and the rest of the code calls
ordinary typed functions.  If a later scipy release widens its stubs, this
module is the one place to delete.
"""

from __future__ import annotations

from typing import Sequence, cast

import numpy as np


def labelled(
    mask: np.ndarray,
    *,
    structure: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """The connected pieces of ``mask``, and how many there are."""

    from scipy.ndimage import label  # noqa: PLC0415

    found = cast(
        "tuple[np.ndarray, int]",
        label(mask, structure=structure),
    )
    return np.asarray(found[0]), int(found[1])


def gaussian_derivative(
    field: np.ndarray,
    sigma: float | Sequence[float],
    *,
    order: tuple[int, int],
) -> np.ndarray:
    """A Gaussian-smoothed derivative of ``field``, ``order`` per axis."""

    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    return np.asarray(
        gaussian_filter(field, sigma, order=order)  # pyright: ignore[reportArgumentType]
    )


__all__ = ["gaussian_derivative", "labelled"]
