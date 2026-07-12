"""GUI-independent cooperative cancellation for bounded artifact computation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias


CancellationProbe: TypeAlias = Callable[[], bool]
DEFAULT_CANCELLATION_POLL_INTERVAL = 256


class ArtifactComputationCancelledError(RuntimeError):
    """A computation stopped at a deterministic, side-effect-free boundary."""


def raise_if_cancelled(cancellation_probe: CancellationProbe | None) -> None:
    """Raise the unique core cancellation signal when cancellation is requested."""

    if cancellation_probe is not None and bool(cancellation_probe()):
        raise ArtifactComputationCancelledError(
            "artifact computation was cancelled at a cooperative boundary"
        )


def poll_cancellation(
    cancellation_probe: CancellationProbe | None,
    iteration: int,
    *,
    interval: int = DEFAULT_CANCELLATION_POLL_INTERVAL,
) -> None:
    """Poll every fixed number of deterministic loop iterations, including zero."""

    if cancellation_probe is None:
        return
    if iteration % interval == 0:
        raise_if_cancelled(cancellation_probe)


__all__ = [
    "ArtifactComputationCancelledError",
    "CancellationProbe",
    "DEFAULT_CANCELLATION_POLL_INTERVAL",
    "poll_cancellation",
    "raise_if_cancelled",
]
