import subprocess
import sys
import unittest


class CliTest(unittest.TestCase):
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "clawdpo", "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Run a ClawDPO training task.", result.stdout)


if __name__ == "__main__":
    unittest.main()
