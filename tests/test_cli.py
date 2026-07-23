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

    def test_shell_entrypoints_parse(self) -> None:
        for script in (
            "infra/cli/curl.sh",
            "infra/cli/database.sh",
            "infra/cli/dpo.sh",
        ):
            result = subprocess.run(
                ["sh", "-n", script],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
