import csv
import fcntl
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from workflow import select_candidates


BRANCH_ARTIFACTS = (
    "candidate-packets.jsonl",
    "branch-localization-tasks.jsonl",
    "branch-localization-results.jsonl",
    "branch-candidates.jsonl",
    "branch-rollouts.jsonl",
    "branch-outcome-tasks.jsonl",
    "branch-outcome-results.jsonl",
    "evaluated-branches.jsonl",
    "verified-branch-points.jsonl",
)


class WorkflowTest(unittest.TestCase):
    def test_branch_stages_advance_manifest_and_guard_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            directory = runs / "iteration-001"
            directory.mkdir()
            (directory / "manifest.json").write_text(
                json.dumps({"behavior_policy": "model-v1", "status": "selected"})
            )
            (directory / "candidate-packets.jsonl").write_text("")

            root = Path(__file__).resolve().parents[1]
            script = root / "workflow/run_iteration.py"
            commands = []

            def fake_run(command, check):
                self.assertTrue(check)
                commands.append(command)
                target = Path(command[1]).name
                action = command[2] if target != "branch_rollout.py" else None
                if target == "locate_branch_points.py" and action == "prepare":
                    Path(command[4]).write_text('{"task_id":"location-1"}\n')
                elif target == "locate_branch_points.py":
                    Path(command[5]).write_text(
                        '{"critic":{"localizable":true}}\n'
                    )
                elif target == "branch_rollout.py":
                    Path(command[3]).write_text('{"branch_id":"branch-1"}\n')
                elif target == "evaluate_branch_points.py" and action == "prepare":
                    Path(command[4]).write_text('{"task_id":"outcome-1"}\n')
                elif target == "evaluate_branch_points.py":
                    Path(command[5]).write_text('{"branch_id":"branch-1"}\n')
                    Path(command[6]).write_text('{"branch_id":"branch-1"}\n')
                return SimpleNamespace(returncode=0)

            def run_stage(*arguments, report_error=False):
                argv = [str(script), *(str(argument) for argument in arguments)]
                with patch.object(
                    sys, "path", [str(script.parent), *sys.path]
                ):
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(subprocess, "run", side_effect=fake_run),
                        patch(
                            "run_report.write_report",
                            side_effect=(
                                RuntimeError("broken report")
                                if report_error
                                else None
                            ),
                        ),
                    ):
                        runpy.run_path(str(script), run_name="__main__")

            draft = runs / "draft.jsonl"
            with self.assertRaisesRegex(
                ValueError, "freeze requires completed branch validation"
            ):
                run_stage("freeze", directory, draft)
            self.assertEqual(commands, [])

            with (directory / ".branch-validation.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                    ValueError, "branch validation is already running"
                ):
                    run_stage("branch-prepare", directory)
            self.assertEqual(commands, [])

            with patch("builtins.print") as warning:
                run_stage("branch-prepare", directory, report_error=True)
            self.assertIn("warning: report update failed", warning.call_args.args[0])
            self.assertIs(warning.call_args.kwargs["file"], sys.stderr)
            branch = json.loads((directory / "manifest.json").read_text())[
                "branch_validation"
            ]
            self.assertEqual(
                (branch["status"], branch["localization_tasks"]),
                ("localization_tasks_ready", 1),
            )
            self.assertEqual(
                set(branch["artifacts"]), set(BRANCH_ARTIFACTS[:2])
            )

            localization_results = runs / "location-results.jsonl"
            localization_results.write_text('{"task_id":"location-1"}\n')
            run_stage(
                "branch-rollout",
                directory,
                localization_results,
                "--seed",
                7,
                "--tensor-parallel-size",
                4,
                "--max-tokens",
                99,
            )
            branch = json.loads((directory / "manifest.json").read_text())[
                "branch_validation"
            ]
            self.assertEqual(
                (
                    branch["status"],
                    branch["localized_branches"],
                    branch["trials_per_token"],
                    branch["outcome_tasks"],
                ),
                ("outcome_tasks_ready", 1, 16, 1),
            )
            self.assertEqual(
                branch["localization_critic"],
                "Codex-as-Critic/Branch Localization Critic",
            )
            self.assertEqual(
                set(branch["artifacts"]), set(BRANCH_ARTIFACTS[:6])
            )

            outcome_results = runs / "outcome-results.jsonl"
            outcome_results.write_text('{"task_id":"outcome-1"}\n')
            run_stage(
                "branch-evaluate",
                directory,
                outcome_results,
                "--min-trials",
                16,
                "--min-gain",
                0.25,
                "--min-pass-rate",
                0.75,
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "selected")
            branch = manifest["branch_validation"]
            self.assertEqual(
                (
                    branch["status"],
                    branch["evaluated_branches"],
                    branch["verified_branches"],
                ),
                ("evaluated", 1, 1),
            )
            self.assertEqual(
                branch["outcome_critic"],
                "Codex-as-Critic/Branch Outcome Critic",
            )
            self.assertEqual(set(branch["artifacts"]), set(BRANCH_ARTIFACTS))
            for name, digest in branch["artifacts"].items():
                self.assertEqual(
                    digest, hashlib.sha256((directory / name).read_bytes()).hexdigest()
                )

            expected_commands = [
                (
                    "workflow/locate_branch_points.py",
                    [
                        "prepare",
                        directory / "candidate-packets.jsonl",
                        directory / "branch-localization-tasks.jsonl",
                        "--model",
                        "model-v1",
                    ],
                ),
                (
                    "workflow/locate_branch_points.py",
                    [
                        "finalize",
                        directory / "candidate-packets.jsonl",
                        directory / "branch-localization-results.jsonl",
                        directory / "branch-candidates.jsonl",
                        "--model",
                        "model-v1",
                    ],
                ),
                (
                    "infra/inference/branch_rollout.py",
                    [
                        directory / "branch-candidates.jsonl",
                        directory / "branch-rollouts.jsonl",
                        "--model",
                        "model-v1",
                        "--trials",
                        "16",
                        "--seed",
                        "7",
                        "--tensor-parallel-size",
                        "4",
                        "--max-tokens",
                        "99",
                    ],
                ),
                (
                    "workflow/evaluate_branch_points.py",
                    [
                        "prepare",
                        directory / "branch-rollouts.jsonl",
                        directory / "branch-outcome-tasks.jsonl",
                    ],
                ),
                (
                    "workflow/evaluate_branch_points.py",
                    [
                        "finalize",
                        directory / "branch-rollouts.jsonl",
                        directory / "branch-outcome-results.jsonl",
                        directory / "evaluated-branches.jsonl",
                        directory / "verified-branch-points.jsonl",
                        "--min-trials",
                        "16",
                        "--min-gain",
                        "0.25",
                        "--min-pass-rate",
                        "0.75",
                    ],
                ),
            ]
            self.assertEqual(len(commands), len(expected_commands))
            for command, (program, arguments) in zip(
                commands, expected_commands, strict=True
            ):
                self.assertEqual(command, [sys.executable, root / program, *arguments])
            for name in BRANCH_ARTIFACTS:
                self.assertTrue((directory / name).is_file(), name)

            (directory / "branch-rollouts.jsonl").write_text("tampered\n")
            command_count = len(commands)
            with self.assertRaisesRegex(
                ValueError,
                "branch artifact missing or changed: branch-rollouts.jsonl",
            ):
                run_stage("freeze", directory, draft)
            self.assertEqual(len(commands), command_count)

    def test_select_records_online_gate_and_validates_its_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rollout_path = directory / "rollouts.jsonl"
            evaluated_path = directory / "evaluated.jsonl"
            candidates_path = directory / "candidates.jsonl"
            request_path = directory / "request.json"
            rollout_row = {
                "trace_id": "trace",
                "messages": [{"role": "user", "content": "question"}],
                "behavior_policy": "model",
                "prompt_token_ids": [81, 82],
                "engine": {
                    "vllm_version": "0.test",
                    "tensor_parallel_size": 8,
                    "logprobs_mode": "raw_logprobs",
                },
                "sampling": {"n": 256, "seed": 0},
                "rollouts": [
                    {
                        "sample_index": index,
                        "response": f"response-{index}",
                        "token_ids": [index],
                        "raw_token_logprobs": [float(-index - 1)],
                        "raw_cumulative_logprob": float(-index - 1),
                        "raw_mean_token_logprob": float(-index - 1),
                        "num_tokens": 1,
                    }
                    for index in range(256)
                ],
            }
            rollout_path.write_text(
                json.dumps(rollout_row) + "\n", encoding="utf-8"
            )
            request_path.write_text(
                json.dumps({"model": "judge-model"}), encoding="utf-8"
            )
            judge_result = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"pass": True, "reason": ""}
                                    )
                                }
                            }
                        ]
                    }
                ),
                stderr="",
            )
            argv = [
                "select_candidates.py",
                str(rollout_path),
                str(evaluated_path),
                str(candidates_path),
                "--request-template",
                str(request_path),
                "--workers",
                "4",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    select_candidates.subprocess,
                    "run",
                    return_value=judge_result,
                ),
            ):
                select_candidates.main()

            evaluated = json.loads(evaluated_path.read_text(encoding="utf-8"))
            packet = json.loads(candidates_path.read_text(encoding="utf-8"))
            gate = packet["correctness_gate"]
            self.assertEqual(evaluated["correctness_gate"], gate)
            self.assertEqual(gate["source"], "machine_judge")
            self.assertEqual(gate["model"], "judge-model")
            self.assertEqual(gate["request_template"], str(request_path.resolve()))
            self.assertEqual(
                gate["request_template_sha256"],
                hashlib.sha256(request_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(Path(gate["judge_prompt"]).is_absolute())
            self.assertEqual(len(gate["judge_prompt_sha256"]), 64)

            del rollout_row["prompt_token_ids"]
            rollout_path.write_text(
                json.dumps(rollout_row) + "\n", encoding="utf-8"
            )
            with (
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(ValueError, "prompt_token_ids"),
            ):
                select_candidates.main()

            rollout_row["prompt_token_ids"] = [81, 82]
            rollout_row["rollouts"][0]["raw_token_logprobs"] = []
            rollout_path.write_text(
                json.dumps(rollout_row) + "\n", encoding="utf-8"
            )
            with (
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(ValueError, "invalid rollout sample"),
            ):
                select_candidates.main()

            rollout_row["rollouts"][0]["raw_token_logprobs"] = [-1.0]
            rollout_path.write_text(
                json.dumps(rollout_row) + "\n", encoding="utf-8"
            )
            request_path.write_text(json.dumps({"model": " "}), encoding="utf-8")
            with (
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(ValueError, "non-empty model"),
            ):
                select_candidates.main()

            request_path.write_text(
                json.dumps({"model": "judge-model"}), encoding="utf-8"
            )
            judge_result.stdout = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"pass": True, "reason": None}
                                )
                            }
                        }
                    ]
                }
            )
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    select_candidates.subprocess,
                    "run",
                    return_value=judge_result,
                ),
                self.assertRaisesRegex(ValueError, "pass and reason"),
            ):
                select_candidates.main()

    def test_select_build_evaluate_and_mine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "iteration"
            directory.mkdir()
            messages = [{"role": "user", "content": "question"}]
            rollouts = []
            factuality = []
            for index in range(256):
                score = float(index - 256)
                response = f"response-{index}"
                rollouts.append(
                    {
                        "sample_index": index,
                        "response": response,
                        "token_ids": [index * 2, index * 2 + 1],
                        "raw_token_logprobs": [score, score],
                        "raw_cumulative_logprob": score * 2,
                        "raw_mean_token_logprob": score,
                        "num_tokens": 2,
                    }
                )
                factuality.append(
                    {
                        "trace_id": "trace-1",
                        "sample_index": index,
                        "judge_model": "judge-model",
                        "pass": index < 240,
                        "reason": "" if index < 240 else "factual error",
                        "judge_input_sha256": hashlib.sha256(
                            json.dumps(
                                {"messages": messages, "response": response},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                    }
                )

            rollout_path = directory / "rollouts.jsonl"
            factuality_path = directory / "factuality.jsonl"
            history_path = directory / "history.jsonl"
            evaluated_path = directory / "evaluated-rollouts.jsonl"
            candidates_path = directory / "candidate-packets.jsonl"
            draft_path = directory / "draft.jsonl"
            dataset_path = directory / "dataset.jsonl"
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "behavior_policy": "model-v1",
                        "status": "rolled_out",
                    }
                )
            )
            rollout_path.write_text(
                json.dumps(
                    {
                        "trace_id": "trace-1",
                        "messages": messages,
                        "behavior_policy": "model-v1",
                        "prompt_token_ids": [101, 102, 103],
                        "engine": {
                            "vllm_version": "0.test",
                            "tensor_parallel_size": 8,
                            "logprobs_mode": "raw_logprobs",
                        },
                        "sampling": {"n": 256, "seed": 0},
                        "rollouts": rollouts,
                    }
                )
                + "\n"
            )
            factuality_path.write_text(
                "".join(json.dumps(row) + "\n" for row in factuality)
            )
            history_path.write_text(
                json.dumps(
                    {
                        "trace_id": "trace-1",
                        "messages": messages
                        + [{"role": "assistant", "content": "old chosen"}],
                        "behavior_policy": "model-v0",
                        "rejected_response": "old rejected",
                        "policy_likelihood": {
                            "model": "model-v1",
                            "chosen": {
                                "raw_cumulative_logprob": -20.0,
                                "raw_mean_token_logprob": -10.0,
                                "num_tokens": 2,
                            },
                        },
                    }
                )
                + "\n"
            )

            stale_factuality = directory / "stale-factuality.jsonl"
            stale_rows = [dict(row) for row in factuality]
            stale_rows[0]["judge_input_sha256"] = "0" * 64
            stale_factuality.write_text(
                "".join(json.dumps(row) + "\n" for row in stale_rows)
            )
            stale_result = subprocess.run(
                [
                    sys.executable,
                    "workflow/select_candidates.py",
                    rollout_path,
                    directory / "stale-evaluated.jsonl",
                    directory / "stale-candidates.jsonl",
                    "--factuality",
                    stale_factuality,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(stale_result.returncode, 0)
            self.assertIn("stale", stale_result.stderr)

            subprocess.run(
                [
                    sys.executable,
                    "workflow/run_iteration.py",
                    "select",
                    directory,
                    "--factuality",
                    factuality_path,
                    "--history",
                    history_path,
                ],
                check=True,
            )
            packet = json.loads(candidates_path.read_text())
            evaluated = json.loads(evaluated_path.read_text())
            self.assertEqual(evaluated["prompt_token_ids"], [101, 102, 103])
            self.assertEqual(packet["prompt_token_ids"], [101, 102, 103])
            self.assertEqual(evaluated["engine"], packet["engine"])
            self.assertEqual(evaluated["sampling"], packet["sampling"])
            self.assertEqual(
                evaluated["correctness_gate"], packet["correctness_gate"]
            )
            gate = packet["correctness_gate"]
            judge_prompt = (
                Path(__file__).resolve().parents[1] / "prompt/事实性检测.md"
            )
            self.assertEqual(gate["source"], "provided_factuality")
            self.assertEqual(gate["model"], "judge-model")
            self.assertEqual(gate["factuality"], str(factuality_path.resolve()))
            self.assertEqual(
                gate["factuality_sha256"],
                hashlib.sha256(factuality_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(gate["judge_prompt"], str(judge_prompt))
            self.assertEqual(
                gate["judge_prompt_sha256"],
                hashlib.sha256(judge_prompt.read_bytes()).hexdigest(),
            )
            evaluated_by_index = {
                sample["sample_index"]: sample for sample in evaluated["rollouts"]
            }
            for candidate in packet["candidates"]:
                source = evaluated_by_index[candidate["sample_index"]]
                self.assertEqual(candidate["token_ids"], source["token_ids"])
                self.assertEqual(
                    candidate["raw_token_logprobs"],
                    source["raw_token_logprobs"],
                )
                self.assertEqual(
                    candidate["judge_input_sha256"],
                    source["judge_input_sha256"],
                )
            kinds = [candidate["candidate_kind"] for candidate in packet["candidates"]]
            self.assertEqual(kinds.count("supported_tail_pass"), 16)
            self.assertEqual(kinds.count("high_fail"), 8)
            self.assertEqual(kinds.count("high_pass"), 8)
            self.assertEqual(len(packet["chosen_history"]), 1)
            manifest = json.loads((directory / "manifest.json").read_text())
            for name in BRANCH_ARTIFACTS[1:]:
                (directory / name).write_text("")
            manifest["branch_validation"] = {
                "status": "evaluated",
                "artifacts": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in BRANCH_ARTIFACTS
                },
            }
            (directory / "manifest.json").write_text(json.dumps(manifest))

            chosen = next(
                candidate["response"]
                for candidate in packet["candidates"]
                if candidate["candidate_kind"] == "supported_tail_pass"
            )
            rejected = next(
                candidate["response"]
                for candidate in packet["candidates"]
                if candidate["candidate_kind"] == "high_fail"
            )
            draft_path.write_text(
                json.dumps(
                    {
                        "trace_id": "trace-1",
                        "messages": messages
                        + [{"role": "assistant", "content": chosen}],
                        "rejected_response": rejected,
                    }
                )
                + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    "workflow/run_iteration.py",
                    "freeze",
                    directory,
                    draft_path,
                ],
                check=True,
            )
            dataset = json.loads(dataset_path.read_text())
            self.assertEqual(dataset["messages"][-1]["content"], chosen)
            self.assertEqual(dataset["rejected_response"], rejected)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "frozen")
            self.assertEqual(manifest["dataset_revision"], str(dataset_path.resolve()))

            fake_bin = directory.parent / "bin"
            fake_bin.mkdir()
            fake_swift = fake_bin / "swift"
            fake_swift.write_text(
                "#!/bin/sh\n"
                "echo \"{'loss': 0.5, 'grad_norm': 1.0, 'rewards/margins': 0.2}\"\n"
                "mkdir -p \"$OUTPUT_DIR\"\n"
            )
            fake_swift.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            subprocess.run(
                [
                    sys.executable,
                    "workflow/run_iteration.py",
                    "train",
                    directory,
                ],
                check=True,
                env=environment,
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "trained")
            self.assertEqual(
                manifest["candidate_model"],
                str((directory / "candidate-model").resolve()),
            )
            test_set_path = directory / "external-test-set.jsonl"
            test_rows = [
                {
                    "trace_id": "test-1",
                    "messages": [{"role": "user", "content": "test question 1"}],
                },
                {
                    "trace_id": "test-2",
                    "messages": [{"role": "user", "content": "test question 2"}],
                },
                {
                    "trace_id": "test-3",
                    "messages": [{"role": "user", "content": "test question 3"}],
                },
            ]
            test_set_path.write_text(
                "".join(json.dumps(row) + "\n" for row in test_rows)
            )
            results_path = directory / "machine-results.jsonl"
            results_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {
                            "trace_id": "test-1",
                            "best_model": "model-v1",
                            "candidate_model": str(
                                (directory / "candidate-model").resolve()
                            ),
                            "best_factuality_pass": True,
                            "candidate_factuality_pass": True,
                            "quality_winner": "candidate",
                        },
                        {
                            "trace_id": "test-2",
                            "best_model": "model-v1",
                            "candidate_model": str(
                                (directory / "candidate-model").resolve()
                            ),
                            "best_factuality_pass": False,
                            "candidate_factuality_pass": True,
                            "quality_winner": "candidate",
                        },
                        {
                            "trace_id": "test-3",
                            "best_model": "model-v1",
                            "candidate_model": str(
                                (directory / "candidate-model").resolve()
                            ),
                            "best_factuality_pass": True,
                            "candidate_factuality_pass": True,
                            "quality_winner": "best",
                        },
                    )
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    "workflow/run_iteration.py",
                    "evaluate",
                    directory,
                    test_set_path,
                    results_path,
                ],
                check=True,
            )
            promotion = json.loads((directory / "promotion.json").read_text())
            self.assertTrue(promotion["promoted"])
            self.assertEqual(promotion["previous_best_model"], "model-v1")
            self.assertEqual(promotion["summary"]["candidate_wins"], 2)
            self.assertIn("rewards/margins", (directory / "training.log").read_text())

            where_path = directory / "where.sql"
            where_path.write_text(
                "WHERE conversation_detail ~* '(test|similar)';\n"
            )
            database_result = directory / "database.csv"
            with database_result.open("w", newline="") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=["create_time", "conversation_detail"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "create_time": "2026-01-01",
                        "conversation_detail": json.dumps(
                            {
                                "messages": test_rows[0]["messages"]
                                + [
                                    {
                                        "role": "assistant",
                                        "content": "original test answer",
                                    }
                                ]
                            }
                        ),
                    }
                )
                writer.writerow(
                    {
                        "create_time": "2026-01-02",
                        "conversation_detail": json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "similar new question"}
                                ]
                            }
                        ),
                    }
                )
            subprocess.run(
                [
                    sys.executable,
                    "workflow/run_iteration.py",
                    "mine",
                    directory,
                    where_path,
                    "--database-result",
                    database_result,
                ],
                check=True,
            )
            with (directory / "mined-train.csv").open(newline="") as source:
                mined = list(csv.DictReader(source))
            self.assertEqual(len(mined), 1)
            self.assertIn("similar new question", mined[0]["conversation_detail"])
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "accepted")
            self.assertEqual(manifest["mining_summary"]["excluded_test_rows"], 1)
            report = (directory.parent / "report.md").read_text()
            self.assertIn("## Training Triples", report)
            self.assertIn(str(dataset_path.resolve()), report)
            self.assertIn(str((directory / "candidate-model").resolve()), report)
            self.assertIn("| iteration | accepted |", report)


if __name__ == "__main__":
    unittest.main()
