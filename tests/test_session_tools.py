import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SessionToolsTest(unittest.TestCase):
    def test_prepare_history_and_prompt_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            iteration = runs / "iteration-001"
            iteration.mkdir(parents=True)
            (iteration / "manifest.json").write_text(
                json.dumps({"behavior_policy": "model-v0", "status": "accepted"})
            )
            (iteration / "dataset.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "train-1",
                        "messages": [
                            {"role": "user", "content": "old question"},
                            {"role": "assistant", "content": "old chosen"},
                        ],
                        "rejected_response": "old rejected",
                    }
                )
                + "\n"
            )

            history = root / "history.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "workflow/prepare_data.py",
                    "history",
                    runs,
                    history,
                ],
                check=True,
            )
            historical = json.loads(history.read_text())
            self.assertEqual(historical["behavior_policy"], "model-v0")
            self.assertEqual(historical["source_iteration"], "iteration-001")

            test_set = root / "test.jsonl"
            test_messages = [{"role": "user", "content": "test question"}]
            test_set.write_text(
                json.dumps({"trace_id": "test-1", "messages": test_messages}) + "\n"
            )
            pool = root / "pool.jsonl"
            pool.write_text(
                json.dumps(
                    {
                        "trace_id": "seed-1",
                        "messages": [{"role": "user", "content": "seed question"}],
                    }
                )
                + "\n"
            )
            mined = root / "mined.csv"
            with mined.open("w", newline="") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=["create_time", "conversation_detail"],
                )
                writer.writeheader()
                for messages in (
                    test_messages
                    + [{"role": "assistant", "content": "test answer"}],
                    [
                        {"role": "user", "content": "new question"},
                        {"role": "assistant", "content": "original answer"},
                    ],
                ):
                    writer.writerow(
                        {
                            "create_time": "2026-01-01",
                            "conversation_detail": json.dumps({"messages": messages}),
                        }
                    )

            output = root / "next-pool.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "workflow/prepare_data.py",
                    "prompts",
                    output,
                    "--test-set",
                    test_set,
                    "--pool",
                    pool,
                    "--mined",
                    mined,
                ],
                check=True,
            )
            prompts = [
                json.loads(line) for line in output.read_text().splitlines() if line
            ]
            self.assertEqual(len(prompts), 2)
            self.assertEqual(
                prompts[-1]["messages"],
                [{"role": "user", "content": "new question"}],
            )
            self.assertTrue(prompts[-1]["trace_id"].startswith("mined-"))

    def test_machine_judge_builds_test_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_set = root / "test.jsonl"
            messages = [{"role": "user", "content": "question"}]
            test_set.write_text(
                json.dumps({"trace_id": "test-1", "messages": messages}) + "\n"
            )
            best = root / "best.jsonl"
            candidate = root / "candidate.jsonl"
            best.write_text(
                json.dumps(
                    {
                        "trace_id": "test-1",
                        "messages": messages,
                        "model": "best-model",
                        "response": "best response",
                    }
                )
                + "\n"
            )
            candidate.write_text(
                json.dumps(
                    {
                        "trace_id": "test-1",
                        "messages": messages,
                        "model": "candidate-model",
                        "response": "candidate response",
                    }
                )
                + "\n"
            )
            factuality_template = root / "factuality.json"
            quality_template = root / "quality.json"
            factuality_template.write_text(json.dumps({"model": "judge"}))
            quality_template.write_text(json.dumps({"model": "judge"}))
            fake_curl = root / "fake-curl.py"
            fake_curl.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "request = json.load(open(sys.argv[1], encoding='utf-8'))\n"
                "system = request['messages'][0]['content']\n"
                "result = ({'pass': True, 'reason': ''} if '事实性检测' in system "
                "else {'winner': 'B', 'reason': 'candidate is better'})\n"
                "print(json.dumps({'choices': [{'message': "
                "{'content': json.dumps(result)}}]}))\n"
            )
            fake_curl.chmod(0o755)

            output = root / "results.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "workflow/evaluate_test.py",
                    "judge",
                    test_set,
                    best,
                    candidate,
                    output,
                    "--factuality-template",
                    factuality_template,
                    "--quality-template",
                    quality_template,
                    "--workers",
                    "1",
                    "--curl",
                    fake_curl,
                ],
                check=True,
            )
            result = json.loads(output.read_text())
            self.assertTrue(result["best_factuality_pass"])
            self.assertTrue(result["candidate_factuality_pass"])
            self.assertEqual(result["quality_winner"], "candidate")
            self.assertEqual(result["candidate_model"], "candidate-model")

    def test_build_and_record_training_failure_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iteration = root / "iteration-001"
            iteration.mkdir()
            pair = {
                "trace_id": "train-1",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "chosen"},
                ],
                "rejected_response": "rejected",
            }
            dataset = iteration / "dataset.jsonl"
            dataset.write_text(json.dumps(pair) + "\n")
            draft = dict(pair)
            draft.update(
                {
                    "pair_type": "quality_lift",
                    "reason": "chosen is clearer",
                }
            )
            (iteration / "draft-pairs.jsonl").write_text(json.dumps(draft) + "\n")
            manifest = {
                "behavior_policy": "base-model",
                "status": "rejected",
                "dataset_revision": str(dataset.resolve()),
                "candidate_model": "candidate-model",
                "evaluation_summary": {
                    "total": 1,
                    "best_factuality_failures": 0,
                    "candidate_factuality_failures": 0,
                    "candidate_wins": 0,
                    "best_wins": 1,
                    "ties": 0,
                    "uncertain": 0,
                },
                "next_best_model": "base-model",
            }
            (iteration / "manifest.json").write_text(json.dumps(manifest))

            for name, model, chosen, rejected in (
                ("base-pair-scores.jsonl", "base-model", -2.0, -2.5),
                ("candidate-pair-scores.jsonl", "candidate-model", -2.2, -3.0),
            ):
                scored = dict(pair)
                scored["policy_likelihood"] = {
                    "model": model,
                    "chosen": {
                        "raw_mean_token_logprob": chosen,
                        "num_tokens": 10,
                    },
                    "rejected": {
                        "raw_mean_token_logprob": rejected,
                        "num_tokens": 8,
                    },
                }
                (iteration / name).write_text(json.dumps(scored) + "\n")

            result = {
                "trace_id": "test-1",
                "best_factuality_pass": True,
                "candidate_factuality_pass": True,
                "quality_winner": "best",
            }
            (iteration / "test-results.jsonl").write_text(json.dumps(result) + "\n")
            (iteration / "test-badcases.jsonl").write_text(json.dumps(result) + "\n")
            (iteration / "training.log").write_text(
                "{'loss': 0.4, 'grad_norm': 1.2, 'rewards/margins': 0.3}\n"
            )

            subprocess.run(
                [
                    sys.executable,
                    "workflow/build_diagnosis_packet.py",
                    iteration,
                ],
                check=True,
            )
            packet = json.loads((iteration / "diagnosis-packet.json").read_text())
            self.assertEqual(packet["pair_summary"]["raw_margin_improved"], 1)
            self.assertEqual(
                packet["pair_summary"]["chosen_down_but_raw_margin_up"],
                1,
            )
            self.assertEqual(
                packet["pair_type_summaries"]["quality_lift"]["pairs"],
                1,
            )

            diagnosis_path = root / "diagnosis.json"
            diagnosis_path.write_text(
                json.dumps(
                    {
                        "verdict": "data_problem",
                        "summary": "chosen quality is too weak",
                        "evidence": ["the Test response regressed"],
                        "data_findings": ["quality_lift pair is weak"],
                        "training_findings": ["margin increased"],
                        "next_attempt": {
                            "base_model": "base-model",
                            "action": "rebuild_dataset",
                            "single_change": "remove weak quality_lift pairs",
                        },
                    }
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    "workflow/run_iteration.py",
                    "diagnose",
                    iteration,
                    diagnosis_path,
                ],
                check=True,
            )
            recorded = json.loads((iteration / "manifest.json").read_text())
            self.assertEqual(recorded["status"], "rejected")
            self.assertEqual(recorded["diagnosis_summary"]["verdict"], "data_problem")
            self.assertIn("data_problem", (root / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
