import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main


class TestMainCLI(unittest.TestCase):
    def test_run_cli_routes_review_command(self):
        with patch("main.review_mesh") as mock_review:
            with patch("sys.argv", ["main.py", "--review", "tile.obj", "review.png"]):
                main.run_cli()

        mock_review.assert_called_once_with("tile.obj", "review.png")

    def test_print_help_mentions_review_command(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main.print_help()

        text = buffer.getvalue()
        self.assertIn("--review <mesh_file>", text)
        self.assertIn("Recording-surface review sheet", text)
        self.assertIn("--generate-synthetic <preset>", text)
        self.assertIn("Synthetic benchmark suite + review sheets", text)

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


if __name__ == "__main__":
    unittest.main()
