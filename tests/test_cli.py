import subprocess
import sys
import unittest


class CliTest(unittest.TestCase):
    def test_python_entrypoints_show_help(self) -> None:
        for script in (
            "infra/inference/rollout.py",
            "infra/inference/rescore.py",
            "workflow/select_candidates.py",
            "workflow/build_pairs.py",
            "workflow/prepare_data.py",
            "workflow/evaluate_test.py",
            "workflow/build_diagnosis_packet.py",
            "workflow/run_report.py",
            "workflow/run_iteration.py",
        ):
            result = subprocess.run(
                [sys.executable, script, "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

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
