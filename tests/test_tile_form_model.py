from src.core.tile_form_model import (
    AxisHint,
    AxisSource,
    MandrelFitResult,
    SectionObservation,
    SplitScheme,
    TileClass,
    TileInterpretationState,
    TileInterpretationSlot,
)


def test_tile_interpretation_state_roundtrip():
    state = TileInterpretationState(
        tile_class=TileClass.SUGKIWA,
        split_scheme=SplitScheme.QUARTER,
        axis_hint=AxisHint(
            source=AxisSource.SELECTED_PATCH_PCA,
            vector_world=(0.0, 1.0, 0.0),
            origin_world=(1.0, 2.0, 3.0),
            confidence=0.82,
            face_count=124,
            note="selected patch",
        ),
        section_observations=[
            SectionObservation(
                station=1.25,
                origin_world=(1.0, 3.25, 3.0),
                normal_world=(0.0, 1.0, 0.0),
                confidence=0.7,
                accepted=True,
                profile_contour_count=1,
                profile_point_count=42,
                profile_width_world=5.2,
                profile_depth_world=3.1,
                profile_radius_median_world=2.45,
                profile_radius_iqr_world=0.18,
                note="candidate",
            )
        ],
        mandrel_fit=MandrelFitResult(
            radius_world=7.5,
            radius_spread_world=0.3,
            axis_origin_world=(1.0, 2.0, 3.0),
            axis_vector_world=(0.0, 1.0, 0.0),
            confidence=0.76,
            used_sections=4,
            used_points=188,
            scope="현재 선택 표면",
            note="rough fit",
        ),
        record_view="top",
        record_strategy="canonical_visible",
        workflow_stage="axis_hint",
        note="draft",
        updated_at_iso="2026-03-20T10:00:00",
    )
    state.save_slot(
        slot_key="slot_1",
        label="상면 초안",
        selected_faces=[8, 2, 4],
    )

    payload = state.to_dict()
    restored = TileInterpretationState.from_dict(payload)

    assert restored.tile_class == TileClass.SUGKIWA
    assert restored.split_scheme == SplitScheme.QUARTER
    assert restored.axis_hint.source == AxisSource.SELECTED_PATCH_PCA
    assert restored.axis_hint.vector_world == (0.0, 1.0, 0.0)
    assert restored.axis_hint.origin_world == (1.0, 2.0, 3.0)
    assert restored.axis_hint.face_count == 124
    assert len(restored.section_observations) == 1
    assert restored.section_observations[0].station == 1.25
    assert restored.section_observations[0].profile_point_count == 42
    assert restored.section_observations[0].profile_radius_median_world == 2.45
    assert restored.mandrel_fit.radius_world == 7.5
    assert restored.mandrel_fit.used_sections == 4
    assert restored.mandrel_fit.scope == "현재 선택 표면"
    assert restored.record_view == "top"
    assert restored.record_strategy == "canonical_visible"
    assert restored.workflow_stage == "axis_hint"
    assert restored.note == "draft"
    assert len(restored.saved_slots) == 1
    assert restored.saved_slots[0].slot_key == "slot_1"
    assert restored.saved_slots[0].selected_faces == [2, 4, 8]
    assert restored.saved_slots[0].label == "상면 초안"


def test_axis_hint_normalizes_direction():
    restored = AxisHint.from_dict(
        {
            "source": "full_mesh_pca",
            "vector_world": [0.0, -10.0, 0.0],
            "origin_world": [0.0, 0.0, 0.0],
            "confidence": 0.4,
            "face_count": 20,
        }
    )

    assert restored.source == AxisSource.FULL_MESH_PCA
    assert restored.vector_world is not None
    assert abs(restored.vector_world[1] - 1.0) < 1e-9


def test_mandrel_fit_result_roundtrip():
    fit = MandrelFitResult(
        radius_world=12.3,
        radius_spread_world=0.4,
        axis_origin_world=(0.0, 0.0, 0.0),
        axis_vector_world=(1.0, 0.0, 0.0),
        confidence=0.88,
        used_sections=5,
        used_points=240,
        scope="전체 메쉬",
        note="demo",
    )

    restored = MandrelFitResult.from_dict(fit.to_dict())

    assert restored.radius_world == 12.3
    assert restored.radius_spread_world == 0.4
    assert restored.axis_vector_world == (1.0, 0.0, 0.0)
    assert restored.used_sections == 5
    assert restored.scope == "전체 메쉬"


def test_tile_interpretation_slot_restore_preserves_saved_slots():
    state = TileInterpretationState(
        tile_class=TileClass.AMKIWA,
        split_scheme=SplitScheme.HALF,
        axis_hint=AxisHint(
            source=AxisSource.FULL_MESH_PCA,
            vector_world=(0.0, 1.0, 0.0),
            origin_world=(0.0, 0.0, 0.0),
            confidence=0.7,
            face_count=88,
        ),
        record_view="bottom",
        record_strategy="canonical_visible",
        workflow_stage="record_surface",
    )
    slot = state.save_slot(slot_key="slot_2", label="하면 기록", selected_faces=[9, 1, 9, 3])
    assert isinstance(slot, TileInterpretationSlot)
    assert slot.selected_faces == [1, 3, 9]

    restored, selected_faces = state.restore_slot("slot_2")

    assert restored.tile_class == TileClass.AMKIWA
    assert restored.split_scheme == SplitScheme.HALF
    assert restored.record_view == "bottom"
    assert restored.workflow_stage == "record_surface"
    assert selected_faces == [1, 3, 9]
    assert len(restored.saved_slots) == 1
    assert restored.saved_slots[0].summary_label().startswith("하면 기록")
