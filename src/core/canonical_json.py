"""Language-neutral canonical JSON values for semantic SHA-256 identities.

RFC 8785 (JCS) is used only where the digest represents JSON semantics rather
than exact container bytes.  Storage files may remain human-readable JSON, but
payload, recipe, and export-claim hashes must be reproducible by independent
implementations in other languages.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import rfc8785


class CanonicalJSONError(ValueError):
    """A value is outside the strict RFC 8785 / I-JSON domain."""


def _plain_json(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth > 100:
        raise CanonicalJSONError(f"JSON value is nested too deeply at {path}")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in value:
            if not isinstance(key, str):
                raise CanonicalJSONError(f"JSON object key at {path} must be a string")
            result[key] = _plain_json(
                value[key],
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise CanonicalJSONError(
        f"unsupported JSON value at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return RFC 8785 canonical bytes, rejecting non-I-JSON numbers."""

    try:
        return rfc8785.dumps(_plain_json(value))
    except (rfc8785.CanonicalizationError, RecursionError, ValueError) as exc:
        raise CanonicalJSONError(f"value cannot be canonicalized as RFC 8785 JSON: {exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CanonicalJSONError",
    "canonical_json_bytes",
    "canonical_json_sha256",
]
