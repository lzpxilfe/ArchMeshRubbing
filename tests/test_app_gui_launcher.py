import unittest
from unittest.mock import patch

import app_gui


class TestAppGuiLauncher(unittest.TestCase):
    def test_main_forwards_to_launch_gui(self):
        with patch("app_gui.launch_gui") as mock_launch:
            app_gui.main()

        mock_launch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
