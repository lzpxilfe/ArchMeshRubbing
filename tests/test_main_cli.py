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


if __name__ == "__main__":
    unittest.main()
