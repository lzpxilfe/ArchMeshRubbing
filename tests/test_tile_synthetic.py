import tempfile
from pathlib import Path

from src.core.tile_form_model import SplitScheme, TileClass
from src.core.tile_synthetic import (
    SyntheticTileGroundTruth,
    SyntheticBenchmarkCaseResult,
    SyntheticBenchmarkSuiteReport,
    TileEvaluationReport,
    default_synthetic_tile_spec,
    evaluate_tile_interpretation,
    generate_synthetic_tile,
    render_synthetic_tile_review_sheet,
    save_synthetic_benchmark_suite,
    save_synthetic_tile_bundle,
    synthetic_tile_spec_from_preset,
)


def test_generate_synthetic_tile_builds_truth_state():
    spec = default_synthetic_tile_spec(
        tile_class=TileClass.SUGKIWA,
        split_scheme=SplitScheme.QUARTER,
        seed=7,
    )
    artifact = generate_synthetic_tile(spec)

    assert artifact.mesh.n_vertices > 0
    assert artifact.mesh.n_faces > 0
    assert artifact.truth.spec.tile_class == TileClass.SUGKIWA
    assert artifact.truth.spec.split_scheme == SplitScheme.QUARTER
    assert artifact.truth.ground_truth_state.axis_hint.is_defined()
    assert artifact.truth.ground_truth_state.mandrel_fit.is_defined()
    assert len(artifact.truth.section_stations) == 5
    assert len(artifact.truth.selected_faces) == artifact.mesh.n_faces


def test_evaluate_tile_interpretation_scores_ground_truth_high():
    spec = default_synthetic_tile_spec(
        tile_class=TileClass.AMKIWA,
        split_scheme=SplitScheme.HALF,
        seed=11,
    )
    artifact = generate_synthetic_tile(spec)
    report = evaluate_tile_interpretation(artifact.truth.ground_truth_state, artifact.truth)

    assert isinstance(report, TileEvaluationReport)
    assert report.tile_class_match is True
    assert report.split_scheme_match is True
    assert report.record_view_match is True
    assert report.overall_score > 0.95


def test_evaluate_tile_interpretation_penalizes_wrong_hypothesis():
    spec = default_synthetic_tile_spec(
        tile_class=TileClass.SUGKIWA,
        split_scheme=SplitScheme.HALF,
        seed=13,
    )
    artifact = generate_synthetic_tile(spec)
    wrong_state = artifact.truth.ground_truth_state.to_dict()
    wrong_state["tile_class"] = "amkiwa"
    wrong_state["split_scheme"] = "quarter"
    wrong_state["record_view"] = "bottom"
    wrong_state["mandrel_fit"]["radius_world"] = float(
        artifact.truth.ground_truth_state.mandrel_fit.radius_world or 0.0
    ) + 18.0
    report = evaluate_tile_interpretation(
        artifact.truth.ground_truth_state.from_dict(wrong_state),
        artifact.truth,
    )

    assert report.tile_class_match is False
    assert report.split_scheme_match is False
    assert report.record_view_match is False
    assert report.overall_score < 0.75


def test_synthetic_truth_and_report_roundtrip():
    artifact = generate_synthetic_tile(
        default_synthetic_tile_spec(
            tile_class=TileClass.SUGKIWA,
            split_scheme=SplitScheme.QUARTER,
            seed=5,
        )
    )
    truth = SyntheticTileGroundTruth.from_dict(artifact.truth.to_dict())
    report = TileEvaluationReport.from_dict(
        evaluate_tile_interpretation(artifact.truth.ground_truth_state, artifact.truth).to_dict()
    )

    assert truth.mesh_name == artifact.truth.mesh_name
    assert truth.ground_truth_state.tile_class == artifact.truth.ground_truth_state.tile_class
    assert report.overall_score > 0.95


def test_synthetic_tile_spec_from_preset_resolves_expected_hypothesis():
    spec = synthetic_tile_spec_from_preset("amkiwa_half", seed=19)

    assert spec.tile_class == TileClass.AMKIWA
    assert spec.split_scheme == SplitScheme.HALF
    assert spec.seed == 19


def test_save_synthetic_tile_bundle_writes_sidecars():
    artifact = generate_synthetic_tile(
        default_synthetic_tile_spec(
            tile_class=TileClass.SUGKIWA,
            split_scheme=SplitScheme.QUARTER,
            seed=23,
        )
    )
    report = evaluate_tile_interpretation(artifact.truth.ground_truth_state, artifact.truth)

    with tempfile.TemporaryDirectory() as td:
        mesh_path = Path(td) / "synthetic_tile.obj"
        paths = save_synthetic_tile_bundle(
            artifact,
            mesh_path,
            interpretation_state=artifact.truth.ground_truth_state,
            evaluation_report=report,
        )

        assert Path(paths["mesh"]).exists()
        assert Path(paths["truth"]).exists()
        assert Path(paths["interpretation"]).exists()
        assert Path(paths["evaluation"]).exists()
        assert Path(paths["review"]).exists()
        assert Path(paths["bundle"]).exists()


def test_save_synthetic_benchmark_suite_writes_summary_files():
    with tempfile.TemporaryDirectory() as td:
        report = save_synthetic_benchmark_suite(
            td,
            presets=("sugkiwa_quarter",),
            seeds=(3,),
            pass_threshold=0.9,
        )

        assert isinstance(report, SyntheticBenchmarkSuiteReport)
        assert report.case_count == 1
        assert report.average_score > 0.95
        assert report.pass_count == 1
        assert report.fail_count == 0
        assert Path(report.cases[0].review_path).exists()
        assert (Path(td) / "synthetic_benchmark_summary.json").exists()
        assert (Path(td) / "synthetic_benchmark_summary.csv").exists()
        summary_csv = (Path(td) / "synthetic_benchmark_summary.csv").read_text(encoding="utf-8")
        assert "pass_threshold" in summary_csv
        assert "0.900000" in summary_csv
        assert "passed" in summary_csv
        assert "review_path" in summary_csv


def test_render_synthetic_tile_review_sheet_builds_combined_image():
    artifact = generate_synthetic_tile(
        default_synthetic_tile_spec(
            tile_class=TileClass.AMKIWA,
            split_scheme=SplitScheme.QUARTER,
            seed=29,
        )
    )
    report = evaluate_tile_interpretation(artifact.truth.ground_truth_state, artifact.truth)
    review = render_synthetic_tile_review_sheet(
        artifact,
        interpretation_state=artifact.truth.ground_truth_state,
        evaluation_report=report,
        dpi=180,
    )

    assert review.combined_image.size[0] > 0
    assert review.combined_image.size[1] > 0


def test_synthetic_benchmark_suite_report_failure_lines():
    report = SyntheticBenchmarkSuiteReport(
        pass_threshold=0.9,
        cases=[
            SyntheticBenchmarkCaseResult(preset="sugkiwa_quarter", seed=1, overall_score=0.96),
            SyntheticBenchmarkCaseResult(preset="amkiwa_half", seed=2, overall_score=0.72),
        ],
        case_count=2,
        pass_count=1,
        fail_count=1,
    )

    lines = report.failing_case_lines(limit=2)
    assert any("FAIL amkiwa_half seed 2" in line for line in lines)
