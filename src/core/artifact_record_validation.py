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
    from .artifact_condition_annotation import (  # noqa: PLC0415
        ArtifactConditionAnnotationError,
        validate_condition_annotation_records,
    )
    from .artifact_crease_record import (  # noqa: PLC0415
        ArtifactCreaseRecordError,
        validate_crease_records,
    )
    from .artifact_developed_rubbing import (  # noqa: PLC0415
        ArtifactDevelopedRubbingError,
        validate_developed_rubbing_records,
    )
    from .artifact_profile_groove import (  # noqa: PLC0415
        ArtifactProfileGrooveError,
        validate_profile_groove_records,
    )
    from .artifact_profile_break import (  # noqa: PLC0415
        ArtifactProfileBreakError,
        validate_profile_break_records,
    )
    from .artifact_paint_cutout import (  # noqa: PLC0415
        ArtifactPaintCutoutError,
        validate_paint_cutout_records,
    )
    from .artifact_relief_shade import (  # noqa: PLC0415
        ArtifactReliefShadeError,
        validate_relief_shade_records,
    )
    from .artifact_technique_annotation import (  # noqa: PLC0415
        ArtifactTechniqueAnnotationError,
        validate_technique_annotation_records,
    )
    from .artifact_texture_lines import (  # noqa: PLC0415
        ArtifactTextureLinesError,
        validate_texture_lines_records,
    )
    from .artifact_rubbing_record import (  # noqa: PLC0415
        ArtifactRubbingRecordError,
        validate_rubbing_records,
    )
    from .artifact_geometry_metrics import (  # noqa: PLC0415
        ArtifactGeometryMetricsError,
        validate_geometry_metrics_records,
    )
    from .artifact_surface_measurement import (  # noqa: PLC0415
        ArtifactSurfaceMeasurementError,
        validate_surface_measurement_records,
    )
    from .artifact_tile_unwrap_record import (  # noqa: PLC0415
        ArtifactTileUnwrapRecordError,
        validate_tile_unwrap_records,
    )
    from .artifact_vector_record import (  # noqa: PLC0415
        ArtifactVectorRecordError,
        validate_vector_records,
    )

    try:
        validate_vector_records(document)
        validate_rubbing_records(document)
        validate_tile_unwrap_records(document)
        validate_geometry_metrics_records(document)
        validate_surface_measurement_records(document)
        validate_condition_annotation_records(document)
        validate_developed_rubbing_records(document)
        validate_profile_groove_records(document)
        validate_profile_break_records(document)
        validate_paint_cutout_records(document)
        validate_relief_shade_records(document)
        validate_technique_annotation_records(document)
        validate_crease_records(document)
        validate_texture_lines_records(document)
    except (
        ArtifactCreaseRecordError,
        ArtifactTextureLinesError,
        ArtifactVectorRecordError,
        ArtifactRubbingRecordError,
        ArtifactTileUnwrapRecordError,
        ArtifactGeometryMetricsError,
        ArtifactSurfaceMeasurementError,
        ArtifactConditionAnnotationError,
        ArtifactDevelopedRubbingError,
        ArtifactProfileGrooveError,
        ArtifactProfileBreakError,
        ArtifactPaintCutoutError,
        ArtifactReliefShadeError,
        ArtifactTechniqueAnnotationError,
    ) as exc:
        raise ArtifactKnownRecordError(str(exc)) from exc


__all__ = ["ArtifactKnownRecordError", "validate_known_records"]
