from __future__ import annotations

import math

import pytest

from src.core.mesh_display import format_mesh_display_values


@pytest.mark.parametrize(
    ("unit", "expected_size", "expected_area"),
    [
        ("mm", "10.0 × 20.0 × 30.0 mm", "1,234.5 mm²"),
        ("CM", "10.0 × 20.0 × 30.0 cm", "1,234.5 cm²"),
        ("m", "10.0 × 20.0 × 30.0 m", "1,234.5 m²"),
    ],
)
def test_display_uses_mesh_unit_without_hard_coded_conversion(
    unit: str,
    expected_size: str,
    expected_area: str,
) -> None:
    result = format_mesh_display_values(
        extents=(10.0, 20.0, 30.0),
        surface_area=1234.5,
        unit=unit,
        surface_area_state="exact",
    )

    assert result.size_text == expected_size
    assert result.area_text == expected_area
    assert result.area_state == "exact"


def test_display_marks_estimated_area_explicitly() -> None:
    result = format_mesh_display_values(
        extents=(1, 2, 3),
        surface_area=9.25,
        unit="mm",
        surface_area_state="estimate",
        decimal_places=2,
    )

    assert result.area_text == "약 9.25 mm² (추정)"
    assert result.area_state == "estimate"


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan, None])
def test_invalid_area_is_never_presented_as_a_measurement(value: object) -> None:
    result = format_mesh_display_values(
        extents=(1, 2, 3),
        surface_area=value,
        unit="mm",
        surface_area_state="exact",
    )

    assert result.area_text == "계산 불가"
    assert result.area_state == "unavailable"


@pytest.mark.parametrize("unit", [None, "", "inch", "unknown"])
def test_unknown_unit_fails_closed_instead_of_guessing(unit: object) -> None:
    result = format_mesh_display_values(
        extents=(1, 2, 3),
        surface_area=4.5,
        unit=unit,
        surface_area_state="exact",
    )

    assert result.size_text == "사용 불가 (단위 미확인)"
    assert result.area_text == "계산 불가 (단위 미확인)"
    assert result.unit is None
    assert result.area_state == "unavailable"


def test_invalid_extents_fail_closed_without_hiding_valid_area() -> None:
    result = format_mesh_display_values(
        extents=(1, -2, 3),
        surface_area=4.5,
        unit="mm",
        surface_area_state="exact",
    )

    assert result.size_text == "사용 불가"
    assert result.area_text == "4.5 mm²"


@pytest.mark.parametrize("decimal_places", [-1, 7, True, 1.5])
def test_decimal_places_are_bounded(decimal_places: object) -> None:
    with pytest.raises(ValueError, match="decimal_places"):
        format_mesh_display_values(
            extents=(1, 2, 3),
            surface_area=4.5,
            unit="mm",
            surface_area_state="exact",
            decimal_places=decimal_places,  # type: ignore[arg-type]
        )
