import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WorkflowTest(unittest.TestCase):
    def test_select_build_evaluate_and_mine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "iteration"
            directory.mkdir()
            messages = [{"role": "user", "content": "question"}]
            rollouts = []
            factuality = []
            for index in range(256):
                score = float(index - 256)
                rollouts.append(
                    {
                        "sample_index": index,
                        "response": f"response-{index}",
                        "raw_cumulative_logprob": score * 2,
                        "raw_mean_token_logprob": score,
                        "num_tokens": 2,
                    }
                )
                factuality.append(
                    {
                        "trace_id": "trace-1",
                        "sample_index": index,
                        "pass": index < 240,
                        "reason": "" if index < 240 else "factual error",
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
            kinds = [candidate["candidate_kind"] for candidate in packet["candidates"]]
            self.assertEqual(kinds.count("supported_tail_pass"), 16)
            self.assertEqual(kinds.count("high_fail"), 8)
            self.assertEqual(kinds.count("high_pass"), 8)
            self.assertEqual(len(packet["chosen_history"]), 1)

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
