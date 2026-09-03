import io
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main
from src.windows_runtime import UnsupportedWindowsRuntimeError


class TestMainCLI(unittest.TestCase):
    def test_gui_launch_rejects_unsupported_runtime_before_qt_import(self):
        with patch(
            "main.require_supported_windows_client_runtime",
            side_effect=UnsupportedWindowsRuntimeError("unsupported-test-runtime"),
        ), patch.dict("sys.modules", {"app_interactive": None}):
            self.assertFalse(main.launch_gui())

    def test_run_cli_routes_review_command(self):
        with patch("main.review_mesh") as mock_review:
            with patch(
                "sys.argv",
                ["main.py", "--review", "tile.obj", "review.png", "--unit", "mm"],
            ):
                main.run_cli()

        mock_review.assert_called_once_with("tile.obj", "review.png", "mm")

    def test_legacy_mesh_commands_refuse_to_guess_the_source_unit(self):
        # A millimetre scan run through the old centimetre default produced a
        # rubbing and a scale bar ten times too large, with no warning.
        for command in ("--flatten", "--review", "--project", "--separate"):
            with self.subTest(command=command):
                with patch("main.review_mesh"), patch("main.flatten_mesh"), patch(
                    "main.project_mesh"
                ), patch("main.separate_mesh"):
                    with patch("sys.argv", ["main.py", command, "tile.obj"]):
                        self.assertEqual(main.run_cli(), 2)

    def test_unit_option_is_validated_before_any_work(self):
        with patch("sys.argv", ["main.py", "--review", "tile.obj", "--unit", "furlong"]):
            self.assertEqual(main.run_cli(), 2)
        with patch("sys.argv", ["main.py", "--review", "tile.obj", "--unit"]):
            self.assertEqual(main.run_cli(), 2)

    def test_unit_option_accepts_the_equals_form(self):
        with patch("main.flatten_mesh") as mock_flatten:
            with patch("sys.argv", ["main.py", "--flatten", "tile.obj", "--unit=cm"]):
                main.run_cli()

        mock_flatten.assert_called_once_with("tile.obj", None, "cm")

    def test_print_help_mentions_review_command(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main.print_help()

        text = buffer.getvalue()
        self.assertIn("--review <mesh_file>", text)
        self.assertIn("Recording-surface review sheet", text)
        self.assertIn("--generate-synthetic <preset>", text)
        self.assertIn("Synthetic benchmark suite + review sheets", text)
        self.assertIn("--field-pilot-review-template REVIEW.json", text)
        self.assertIn("--field-pilot PROJECT.amr SURVEY.amr-survey", text)
        self.assertIn("--verify-field-pilot PILOT.json", text)

    def test_run_cli_routes_generate_synthetic_command(self):
        with patch("main.generate_synthetic_bundle") as mock_generate:
            with patch("sys.argv", ["main.py", "--generate-synthetic", "sugkiwa_quarter", "7", "synthetic.obj"]):
                main.run_cli()

        mock_generate.assert_called_once_with("sugkiwa_quarter", seed=7, output_path="synthetic.obj")

    def test_run_cli_routes_open_mesh_command(self):
        with patch("main.launch_gui") as mock_launch:
            with patch("sys.argv", ["main.py", "--open-mesh", "tile.obj"]):
                main.run_cli()

        mock_launch.assert_called_once_with(open_mesh="tile.obj")

    def test_run_cli_existing_path_opens_gui(self):
        with patch("main.launch_gui") as mock_launch:
            with patch("main.os.path.exists", return_value=True):
                with patch("sys.argv", ["main.py", "tile.obj"]):
                    main.run_cli()

        mock_launch.assert_called_once_with(open_mesh="tile.obj")

    def test_run_cli_routes_benchmark_synthetic_command(self):
        with patch("main.benchmark_synthetic_tiles") as mock_benchmark:
            with patch("sys.argv", ["main.py", "--benchmark-synthetic", "benchmarks", "1,2,3"]):
                main.run_cli()

        mock_benchmark.assert_called_once_with("benchmarks", seeds_arg="1,2,3")

    def test_run_cli_routes_field_pilot_template_before_logging(self):
        with patch(
            "src.core.field_pilot.write_field_pilot_review_template",
            return_value=SimpleNamespace(warning_message=None),
        ) as write_template:
            with patch("sys.argv", ["main.py", "--field-pilot-review-template", "review.json"]):
                result = main.run_cli()

        self.assertEqual(result, 0)
        write_template.assert_called_once_with("review.json")

    def test_run_cli_routes_verified_field_pilot_and_all_inputs(self):
        report = {"outcome": {"pilot": "verified"}}
        with (
            patch(
                "src.core.field_pilot.build_field_pilot_report",
                return_value=report,
            ) as build_report,
            patch(
                "src.core.field_pilot.write_field_pilot_report",
                return_value=SimpleNamespace(warning_message=None),
            ) as write_report,
            patch(
                "sys.argv",
                [
                    "main.py",
                    "--field-pilot",
                    "artifact.amr",
                    "artifact.amr-survey",
                    "--review",
                    "review.json",
                    "--opengl-report",
                    "opengl.json",
                    "--report",
                    "pilot.json",
                ],
            ),
        ):
            result = main.run_cli()

        self.assertEqual(result, 0)
        build_report.assert_called_once_with(
            "artifact.amr",
            "artifact.amr-survey",
            review="review.json",
            opengl_report="opengl.json",
        )
        write_report.assert_called_once_with("pilot.json", report)

    def test_run_cli_returns_one_for_an_incomplete_published_pilot(self):
        report = {"outcome": {"pilot": "incomplete"}}
        with (
            patch(
                "src.core.field_pilot.build_field_pilot_report",
                return_value=report,
            ),
            patch(
                "src.core.field_pilot.write_field_pilot_report",
                return_value=SimpleNamespace(warning_message=None),
            ),
            patch(
                "sys.argv",
                [
                    "main.py",
                    "--field-pilot",
                    "artifact.amr",
                    "artifact.amr-survey",
                    "--report",
                    "pilot.json",
                ],
            ),
        ):
            result = main.run_cli()

        self.assertEqual(result, 1)

    def test_run_cli_routes_field_pilot_verification_receipt(self):
        receipt = {"ok": True, "evidence": {"pilot_outcome": "incomplete"}}
        with (
            patch(
                "src.core.field_pilot.build_field_pilot_verification_report",
                return_value=receipt,
            ) as verify,
            patch("main.build_info.write_json_report") as write_receipt,
            patch(
                "sys.argv",
                [
                    "main.py",
                    "--verify-field-pilot",
                    "pilot.json",
                    "--report",
                    "receipt.json",
                ],
            ),
        ):
            result = main.run_cli()

        self.assertEqual(result, 0)
        verify.assert_called_once_with("pilot.json")
        write_receipt.assert_called_once_with("receipt.json", receipt)

    def test_run_cli_rejects_duplicate_field_pilot_options(self):
        with patch(
            "sys.argv",
            [
                "main.py",
                "--field-pilot",
                "artifact.amr",
                "artifact.amr-survey",
                "--report",
                "first.json",
                "--report",
                "second.json",
            ],
        ):
            result = main.run_cli()

        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
