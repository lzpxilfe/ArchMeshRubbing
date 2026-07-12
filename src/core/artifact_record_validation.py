"""Registry boundary for strict validation of known DerivedRecord types."""

from __future__ import annotations

from .artifact_document import ArtifactDocument


class ArtifactKnownRecordError(ValueError):
    """A known record type failed its type-specific durable contract."""


def validate_known_records(document: ArtifactDocument) -> None:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactKnownRecordError("document must be an ArtifactDocument")
    # Local imports keep record implementations independent and avoid cycles
    # through ArtifactSession during Digital Rubbing computation.
    from .artifact_rubbing_record import (  # noqa: PLC0415
        ArtifactRubbingRecordError,
        validate_rubbing_records,
    )
    from .artifact_vector_record import (  # noqa: PLC0415
        ArtifactVectorRecordError,
        validate_vector_records,
    )

    try:
        validate_vector_records(document)
        validate_rubbing_records(document)
    except (ArtifactVectorRecordError, ArtifactRubbingRecordError) as exc:
        raise ArtifactKnownRecordError(str(exc)) from exc


__all__ = ["ArtifactKnownRecordError", "validate_known_records"]
