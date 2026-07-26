import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = "workflow/evaluate_branch_points.py"
ROOT = Path(__file__).resolve().parents[1]


def content_id(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class BranchEvaluationTest(unittest.TestCase):
    def test_jsonl_reader_is_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text('{"first":true}\nnot-json\n')
            iterator = runpy.run_path(SCRIPT)["iter_jsonl"](path)

            self.assertEqual(next(iterator), {"first": True})
            with self.assertRaises(ValueError):
                next(iterator)

    def make_input(self, directory: Path, trials: int = 3) -> Path:
        messages = [{"role": "user", "content": "question"}]
        source_response = "source response"
        source_reason = "wrong fact"
        prefix = [1, 2, 3, 4]
        rollouts = []
        for arm, token_id, label in (
            ("original_token", 7, "original"),
            ("alternative_token", 11, "alternative"),
        ):
            for replica in range(trials):
                rollouts.append(
                    {
                        "arm": arm,
                        "replica": replica,
                        "branch_token_id": token_id,
                        "branch_token_text": str(token_id),
                        "branch_token_raw_logprob": -0.5,
                        "response_token_ids": prefix
                        + [token_id, 100 + replica],
                        "response": f"{label} response {replica}",
                        "finish_reason": "stop",
                        "stop_reason": None,
                    }
                )
        row = {
            "branch_id": "trace-1:3:4",
            "trace_id": "trace-1",
            "messages": messages,
            "behavior_policy": "model-a",
            "source_sample_index": 3,
            "source_response": source_response,
            "source_response_token_ids": prefix + [7, 8],
            "source_likelihood_region": "high",
            "source_correctness_pass": False,
            "source_correctness_reason": source_reason,
            "source_judge_input_sha256": content_id(
                {"messages": messages, "response": source_response}
            ),
            "critic": {"localizable": True, "reason": "possible error"},
            "critic_input_sha256": content_id(
                {
                    "messages": messages,
                    "response": source_response,
                    "correctness_reason": source_reason,
                }
            ),
            "pivot_index": 4,
            "bad_token_id": 7,
            "bad_token_logprob": -0.5,
            "prompt_token_ids": [90, 91],
            "sampling": {
                "trials": trials,
                "selection_seed": 9,
                "continuation_seed": 9 + trials,
            },
            "alternative_token_id": 11,
            "alternative_token_text": "11",
            "alternative_token_raw_logprob": -0.5,
            "source_engine": "opaque provenance",
            "future_provenance": {"must_survive": True},
            "rollouts": rollouts,
        }
        input_path = directory / "branches.jsonl"
        input_path.write_text(json.dumps(row) + "\n")
        return input_path

    def run_prepare(
        self,
        input_path: Path,
        tasks_path: Path,
        *extra: str,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "prepare",
                input_path,
                tasks_path,
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def write_results(
        self,
        path: Path,
        tasks_path: Path,
        original: list[tuple[bool, bool]],
        alternative: list[tuple[bool, bool]],
        winner: str,
    ) -> list[dict]:
        tasks = [
            json.loads(line) for line in tasks_path.read_text().splitlines()
        ]
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        group_ids = {}
        group_results = []
        for group in task["groups"]:
            response_results = []
            label = None
            for response in group["responses"]:
                parts = response["response"].split()
                label = parts[0]
                replica = int(parts[-1])
                factuality_pass, task_pass = {
                    "original": original,
                    "alternative": alternative,
                }[label][replica]
                response_results.append(
                    {
                        "response_id": response["response_id"],
                        "factuality_pass": factuality_pass,
                        "task_pass": task_pass,
                        "reason": f"{label} reason {replica}",
                    }
                )
            group_ids[label] = group["group_id"]
            group_results.append(
                {
                    "group_id": group["group_id"],
                    "responses": response_results,
                }
            )
        if winner in group_ids:
            winner = group_ids[winner]
        rows = [
            {
                "task_id": task["task_id"],
                "groups": group_results,
                "group_winner": winner,
                "group_reason": "group quality reason",
            }
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows

    def run_finalize(
        self,
        input_path: Path,
        results_path: Path,
        evaluated: Path,
        verified: Path,
        *extra: str,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "finalize",
                input_path,
                results_path,
                evaluated,
                verified,
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def test_prepare_outputs_only_anonymous_content_bound_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = self.make_input(directory, trials=16)
            tasks_path = directory / "tasks.jsonl"
            result = self.run_prepare(input_path, tasks_path)
            self.assertEqual(result.returncode, 0, result.stderr)

            tasks = [
                json.loads(line) for line in tasks_path.read_text().splitlines()
            ]
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(set(task), {"task_id", "messages", "groups"})
            self.assertEqual(len(task["task_id"]), 64)
            self.assertEqual(len(task["groups"]), 2)
            self.assertEqual(
                [group["group_id"] for group in task["groups"]],
                sorted(group["group_id"] for group in task["groups"]),
            )
            for group in task["groups"]:
                self.assertEqual(set(group), {"group_id", "responses"})
                self.assertEqual(len(group["group_id"]), 64)
                self.assertEqual(len(group["responses"]), 16)
                self.assertTrue(
                    all(
                        set(response) == {"response_id", "response"}
                        and len(response["response_id"]) == 64
                        for response in group["responses"]
                    )
                )
            self.assertEqual(
                sum(len(group["responses"]) for group in task["groups"]),
                32,
            )
            serialized = tasks_path.read_text()
            self.assertNotIn("trace-1", serialized)
            self.assertNotIn('"arm"', serialized)
            self.assertNotIn("branch_token", serialized)
            self.assertNotIn("possible error", serialized)

            repeated = directory / "tasks-repeated.jsonl"
            repeated_result = self.run_prepare(input_path, repeated)
            self.assertEqual(repeated_result.returncode, 0, repeated_result.stderr)
            self.assertEqual(tasks_path.read_text(), repeated.read_text())

            prompt_specs = (
                ("--codex-prompt", ROOT / "prompt/codex/分叉结果评测.md"),
                ("--factuality-prompt", ROOT / "prompt/事实性检测.md"),
                ("--quality-prompt", ROOT / "prompt/回复竞对.md"),
            )
            original_ids = [task["task_id"] for task in tasks]
            for index, (flag, source) in enumerate(prompt_specs):
                changed_prompt = directory / f"prompt-{index}.md"
                changed_prompt.write_bytes(source.read_bytes() + b"\nchanged\n")
                changed_tasks = directory / f"tasks-{index}.jsonl"
                changed_result = self.run_prepare(
                    input_path,
                    changed_tasks,
                    flag,
                    str(changed_prompt),
                )
                self.assertEqual(changed_result.returncode, 0, changed_result.stderr)
                changed_ids = [
                    json.loads(line)["task_id"]
                    for line in changed_tasks.read_text().splitlines()
                ]
                self.assertNotEqual(changed_ids, original_ids)

            original_row = json.loads(input_path.read_text())
            evidence_changes = (
                ("behavior", {"behavior_policy": "model-b"}),
                ("engine", {"source_engine": {"version": "different"}}),
                (
                    "sampling",
                    {
                        "sampling": {
                            **original_row["sampling"],
                            "selection_seed": 10,
                        }
                    },
                ),
            )
            for label, change in evidence_changes:
                changed_input = directory / f"branches-{label}.jsonl"
                changed_input.write_text(json.dumps({**original_row, **change}) + "\n")
                changed_tasks = directory / f"tasks-{label}.jsonl"
                changed_result = self.run_prepare(changed_input, changed_tasks)
                self.assertEqual(changed_result.returncode, 0, changed_result.stderr)
                changed_id = json.loads(changed_tasks.read_text())["task_id"]
                self.assertNotEqual(changed_id, original_ids[0])

    def test_finalize_summarizes_both_checks_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = self.make_input(directory)
            tasks_path = directory / "tasks.jsonl"
            self.assertEqual(
                self.run_prepare(input_path, tasks_path).returncode,
                0,
            )
            results_path = directory / "results.jsonl"
            self.write_results(
                results_path,
                tasks_path,
                [(False, True), (True, True), (False, True)],
                [(True, True), (True, False), (True, True)],
                "alternative",
            )
            evaluated = directory / "evaluated.jsonl"
            verified = directory / "verified.jsonl"
            result = self.run_finalize(
                input_path,
                results_path,
                evaluated,
                verified,
                "--min-trials",
                "3",
                "--min-gain",
                "0.3",
                "--min-pass-rate",
                "0.6",
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            row = json.loads(evaluated.read_text())
            summary = row["summary"]
            self.assertEqual(
                summary["original_token"]["factuality"],
                {"passes": 1, "trials": 3, "rate": 1 / 3},
            )
            self.assertEqual(
                summary["original_token"]["task_completion"],
                {"passes": 3, "trials": 3, "rate": 1.0},
            )
            self.assertEqual(
                summary["original_token"]["overall"],
                {"passes": 1, "trials": 3, "rate": 1 / 3},
            )
            self.assertEqual(
                summary["alternative_token"]["overall"],
                {"passes": 2, "trials": 3, "rate": 2 / 3},
            )
            self.assertAlmostEqual(summary["overall_gain"], 1 / 3)
            self.assertEqual(
                summary["paired_counts"],
                {"00": 0, "01": 2, "10": 1, "11": 0},
            )
            self.assertEqual(
                summary["quality"],
                {
                    "winner": "alternative_token",
                    "reason": "group quality reason",
                },
            )
            self.assertTrue(summary["factuality_not_lower"])
            self.assertTrue(summary["alternative_group_wins"])
            self.assertTrue(summary["verified"])
            self.assertEqual(row["gate"]["source"], "codex_as_critic")
            self.assertEqual(set(row["gate"]["prompts"]), {"codex", "factuality", "quality"})
            for name, path in (
                ("codex", ROOT / "prompt/codex/分叉结果评测.md"),
                ("factuality", ROOT / "prompt/事实性检测.md"),
                ("quality", ROOT / "prompt/回复竞对.md"),
            ):
                self.assertEqual(
                    row["gate"]["prompts"][name]["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            self.assertEqual(row["source_engine"], "opaque provenance")
            self.assertTrue(
                all(
                    rollout["codex_evaluation"]["overall_pass"]
                    == (
                        rollout["codex_evaluation"]["factuality_pass"]
                        and rollout["codex_evaluation"]["task_pass"]
                    )
                    for rollout in row["rollouts"]
                )
            )

            verified_row = json.loads(verified.read_text())
            self.assertNotIn("rollouts", verified_row)
            self.assertNotIn("chosen", verified_row)
            self.assertNotIn("rejected", verified_row)
            self.assertEqual(
                verified_row["future_provenance"],
                {"must_survive": True},
            )

    def test_factuality_and_quality_are_both_hard_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = self.make_input(directory)
            tasks_path = directory / "tasks.jsonl"
            self.assertEqual(
                self.run_prepare(input_path, tasks_path).returncode,
                0,
            )
            cases = (
                (
                    "factuality",
                    [(True, False)] * 3,
                    [(True, True), (True, True), (False, True)],
                    "alternative",
                    "factuality_not_lower",
                ),
                (
                    "quality",
                    [(False, True)] * 3,
                    [(True, True)] * 3,
                    "tie",
                    "alternative_group_wins",
                ),
            )
            for label, original, alternative, winner, failed_check in cases:
                results_path = directory / f"results-{label}.jsonl"
                self.write_results(
                    results_path,
                    tasks_path,
                    original,
                    alternative,
                    winner,
                )
                evaluated = directory / f"evaluated-{label}.jsonl"
                verified = directory / f"verified-{label}.jsonl"
                result = self.run_finalize(
                    input_path,
                    results_path,
                    evaluated,
                    verified,
                    "--min-trials",
                    "3",
                    "--min-gain",
                    "0.5",
                    "--min-pass-rate",
                    "0.6",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(evaluated.read_text())["summary"]
                self.assertFalse(summary[failed_check])
                self.assertFalse(summary["verified"])
                self.assertEqual(verified.read_text(), "")

    def test_prepare_rejects_bad_branch_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = self.make_input(directory, trials=2)
            original = json.loads(input_path.read_text())
            cases = []

            stale_hash = json.loads(json.dumps(original))
            stale_hash["source_judge_input_sha256"] = "0" * 64
            cases.append(stale_hash)

            wrong_pivot = json.loads(json.dumps(original))
            wrong_pivot["source_response_token_ids"][4] = 8
            cases.append(wrong_pivot)

            old_arm = json.loads(json.dumps(original))
            old_arm["rollouts"][0]["arm"] = "bad_token"
            cases.append(old_arm)

            missing_replica = json.loads(json.dumps(original))
            missing_replica["rollouts"] = missing_replica["rollouts"][:-1]
            cases.append(missing_replica)

            varying_alternative = json.loads(json.dumps(original))
            varying_alternative["rollouts"][-1]["branch_token_id"] = 12
            varying_alternative["rollouts"][-1]["response_token_ids"][4] = 12
            cases.append(varying_alternative)

            wrong_top_alternative_id = json.loads(json.dumps(original))
            wrong_top_alternative_id["alternative_token_id"] = 12
            cases.append(wrong_top_alternative_id)

            wrong_top_alternative_text = json.loads(json.dumps(original))
            wrong_top_alternative_text["alternative_token_text"] = "twelve"
            cases.append(wrong_top_alternative_text)

            wrong_top_alternative_logprob = json.loads(json.dumps(original))
            wrong_top_alternative_logprob["alternative_token_raw_logprob"] = -0.4
            cases.append(wrong_top_alternative_logprob)

            wrong_top_original_logprob = json.loads(json.dumps(original))
            wrong_top_original_logprob["bad_token_logprob"] = -0.4
            cases.append(wrong_top_original_logprob)

            wrong_rollout_logprob = json.loads(json.dumps(original))
            wrong_rollout_logprob["rollouts"][0]["branch_token_raw_logprob"] = -0.4
            cases.append(wrong_rollout_logprob)

            invalid_rollout_logprob = json.loads(json.dumps(original))
            invalid_rollout_logprob["rollouts"][0]["branch_token_raw_logprob"] = float(
                "nan"
            )
            cases.append(invalid_rollout_logprob)

            invalid_rollout_text = json.loads(json.dumps(original))
            invalid_rollout_text["rollouts"][0]["branch_token_text"] = None
            cases.append(invalid_rollout_text)

            extra_rollout_field = json.loads(json.dumps(original))
            extra_rollout_field["rollouts"][0]["hidden"] = True
            cases.append(extra_rollout_field)

            duplicate_branch = [original, original]
            for index, rows in enumerate([*[[row] for row in cases], duplicate_branch]):
                changed_input = directory / f"invalid-{index}.jsonl"
                changed_input.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                result = self.run_prepare(
                    changed_input,
                    directory / f"tasks-invalid-{index}.jsonl",
                )
                self.assertNotEqual(result.returncode, 0, index)

    def test_finalize_requires_exact_complete_codex_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = self.make_input(directory, trials=2)
            tasks_path = directory / "tasks.jsonl"
            self.assertEqual(
                self.run_prepare(input_path, tasks_path).returncode,
                0,
            )
            valid_results_path = directory / "results.jsonl"
            valid_rows = self.write_results(
                valid_results_path,
                tasks_path,
                [(False, True)] * 2,
                [(True, True)] * 2,
                "alternative",
            )

            cases = []
            cases.append(valid_rows[:-1])
            cases.append(valid_rows + [valid_rows[0]])
            unknown_task = json.loads(json.dumps(valid_rows))
            unknown_task[0]["task_id"] = "0" * 64
            cases.append(unknown_task)
            extra_field = json.loads(json.dumps(valid_rows))
            extra_field[0]["extra"] = True
            cases.append(extra_field)
            missing_response = json.loads(json.dumps(valid_rows))
            missing_response[0]["groups"][0]["responses"] = (
                missing_response[0]["groups"][0]["responses"][:-1]
            )
            cases.append(missing_response)
            wrong_response = json.loads(json.dumps(valid_rows))
            wrong_response[0]["groups"][0]["responses"][0]["response_id"] = "0" * 64
            cases.append(wrong_response)
            wrong_boolean = json.loads(json.dumps(valid_rows))
            wrong_boolean[0]["groups"][0]["responses"][0]["task_pass"] = "true"
            cases.append(wrong_boolean)
            wrong_winner = json.loads(json.dumps(valid_rows))
            wrong_winner[0]["group_winner"] = "original_token"
            cases.append(wrong_winner)

            for index, rows in enumerate(cases):
                results_path = directory / f"invalid-results-{index}.jsonl"
                results_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                result = self.run_finalize(
                    input_path,
                    results_path,
                    directory / f"evaluated-{index}.jsonl",
                    directory / f"verified-{index}.jsonl",
                )
                self.assertNotEqual(result.returncode, 0, index)

    def test_finalize_rejects_results_from_older_branch_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = self.make_input(directory, trials=2)
            tasks_path = directory / "tasks.jsonl"
            self.assertEqual(
                self.run_prepare(input_path, tasks_path).returncode,
                0,
            )
            results_path = directory / "results.jsonl"
            self.write_results(
                results_path,
                tasks_path,
                [(False, True)] * 2,
                [(True, True)] * 2,
                "alternative",
            )

            changed_row = json.loads(input_path.read_text())
            changed_row["source_engine"] = {"version": "new run"}
            changed_input = directory / "changed-branches.jsonl"
            changed_input.write_text(json.dumps(changed_row) + "\n")
            result = self.run_finalize(
                changed_input,
                results_path,
                directory / "evaluated.jsonl",
                directory / "verified.jsonl",
            )
            self.assertNotEqual(result.returncode, 0)

    def test_empty_input_is_a_valid_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "empty.jsonl"
            tasks_path = directory / "tasks.jsonl"
            results_path = directory / "results.jsonl"
            evaluated = directory / "evaluated.jsonl"
            verified = directory / "verified.jsonl"
            input_path.write_text("")
            results_path.write_text("")

            prepared = self.run_prepare(input_path, tasks_path)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            self.assertEqual(tasks_path.read_text(), "")
            finalized = self.run_finalize(
                input_path,
                results_path,
                evaluated,
                verified,
            )
            self.assertEqual(finalized.returncode, 0, finalized.stderr)
            self.assertEqual(evaluated.read_text(), "")
            self.assertEqual(verified.read_text(), "")

    def test_legacy_machine_judge_flags_are_gone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "empty.jsonl"
            tasks_path = directory / "tasks.jsonl"
            input_path.write_text("")
            result = self.run_prepare(
                input_path,
                tasks_path,
                "--request-template",
                "request.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unrecognized arguments", result.stderr)


if __name__ == "__main__":
    unittest.main()
