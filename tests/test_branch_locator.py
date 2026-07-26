import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from workflow import locate_branch_points


class FakeTokenizer:
    pieces = {10: "前", 11: "错误", 12: "中", 13: "错误", 14: "后", 99: "<eos>"}

    def decode(self, token_ids, skip_special_tokens=False, **_):
        return "".join(
            ""
            if token_id == 99 and skip_special_tokens
            else self.pieces[token_id]
            for token_id in token_ids
        )


class ByteSplitTokenizer:
    prefixes = {
        (): "",
        (20,): "前",
        (20, 21): "前�",
        (20, 21, 22): "前�",
        (20, 21, 22, 23): "前😀",
        (20, 21, 22, 23, 24): "前😀后",
    }

    def decode(self, token_ids, **_):
        return self.prefixes[tuple(token_ids)]


class BranchLocatorTest(unittest.TestCase):
    def make_candidate(self, messages, overrides=None):
        overrides = overrides or {}
        candidate = {
            "candidate_kind": "high_fail",
            "sample_index": 7,
            "response": "前错误中错误后",
            "token_ids": [10, 11, 12, 13, 14, 99],
            "raw_token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
            "raw_cumulative_logprob": -2.1,
            "raw_mean_token_logprob": -0.35,
            "num_tokens": 6,
            "likelihood_region": "high",
            "correctness_pass": False,
            "correctness_reason": "第二处错误",
        }
        candidate.update(overrides)
        if "judge_input_sha256" not in overrides:
            judge_input = json.dumps(
                {"messages": messages, "response": candidate["response"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            candidate["judge_input_sha256"] = hashlib.sha256(
                judge_input.encode()
            ).hexdigest()
        return candidate

    def make_packet(
        self,
        *,
        behavior_policy="model",
        messages=None,
        candidate_overrides=None,
        packet_overrides=None,
    ):
        messages = messages or [{"role": "user", "content": "问题"}]
        packet = {
            "trace_id": "trace",
            "messages": messages,
            "behavior_policy": behavior_policy,
            "prompt_token_ids": [101, 102, 103],
            "engine": {"vllm_version": "test", "tensor_parallel_size": 8},
            "sampling": {"n": 256, "seed": 0},
            "correctness_gate": {
                "source": "provided_factuality",
                "model": "replay",
            },
            "candidates": [
                {"candidate_kind": "high_pass", "sample_index": 999},
                self.make_candidate(messages, candidate_overrides),
            ],
        }
        packet.update(packet_overrides or {})
        return packet

    def call_main(self, argv, tokenizer=None):
        load_tokenizer = Mock(
            return_value=tokenizer if tokenizer is not None else FakeTokenizer()
        )
        with (
            patch.object(sys, "argv", ["locate_branch_points.py", *map(str, argv)]),
            patch.dict(
                sys.modules,
                {
                    "transformers": SimpleNamespace(
                        AutoTokenizer=SimpleNamespace(
                            from_pretrained=load_tokenizer
                        )
                    )
                },
            ),
        ):
            locate_branch_points.main()
        return load_tokenizer

    def run_locator(
        self,
        critic,
        *,
        behavior_policy="model",
        messages=None,
        candidate_overrides=None,
        packet_overrides=None,
        tokenizer=None,
        result_task_id=None,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidates = directory / "candidates.jsonl"
            tasks = directory / "tasks.jsonl"
            results = directory / "results.jsonl"
            output = directory / "output.jsonl"
            packet = self.make_packet(
                behavior_policy=behavior_policy,
                messages=messages,
                candidate_overrides=candidate_overrides,
                packet_overrides=packet_overrides,
            )
            candidates.write_text(
                json.dumps(packet, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            self.call_main(
                ["prepare", candidates, tasks, "--model", "model"],
                tokenizer,
            )
            task = json.loads(tasks.read_text(encoding="utf-8"))
            results.write_text(
                json.dumps(
                    {
                        "task_id": (
                            result_task_id
                            if result_task_id is not None
                            else task["task_id"]
                        ),
                        **critic,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            load_tokenizer = self.call_main(
                ["finalize", candidates, results, output, "--model", "model"],
                tokenizer,
            )
            row = json.loads(output.read_text(encoding="utf-8"))
            if critic["localizable"]:
                load_tokenizer.assert_called_once_with(
                    "model", trust_remote_code=True
                )
            else:
                load_tokenizer.assert_not_called()
            return row, task

    def test_prepare_writes_only_codex_task_fields(self):
        critic = {"localizable": False, "reason": "多个错误交织"}
        row, task = self.run_locator(critic)

        self.assertEqual(
            set(task),
            {"task_id", "messages", "response", "correctness_reason"},
        )
        self.assertEqual(task["messages"], [{"role": "user", "content": "问题"}])
        self.assertEqual(task["response"], "前错误中错误后")
        self.assertEqual(task["correctness_reason"], "第二处错误")
        self.assertEqual(len(task["task_id"]), 64)
        self.assertEqual(row["critic_task_id"], task["task_id"])

    def test_finalize_maps_span_to_original_token(self):
        row, _ = self.run_locator(
            {
                "localizable": True,
                "error_span": "错误",
                "start_char": 4,
                "end_char": 6,
                "error_type": "错误事实",
                "reason": "第二处才是分叉",
            }
        )
        self.assertEqual(row["pivot_index"], 3)
        self.assertEqual(row["bad_token_id"], 13)
        self.assertEqual(row["bad_token_logprob"], -0.4)
        self.assertEqual(row["branch_id"], "trace:7:3")
        self.assertEqual(
            row["source_response_token_ids"], [10, 11, 12, 13, 14, 99]
        )
        self.assertNotIn("source_raw_token_logprobs", row)
        self.assertEqual(row["source_raw_mean_token_logprob"], -0.35)
        self.assertEqual(row["source_likelihood_region"], "high")
        self.assertFalse(row["source_correctness_pass"])
        self.assertEqual(row["prompt_token_ids"], [101, 102, 103])
        self.assertEqual(
            row["source_engine"],
            {"vllm_version": "test", "tensor_parallel_size": 8},
        )
        self.assertEqual(row["source_sampling"], {"n": 256, "seed": 0})

    def test_keeps_non_localizable_result_and_codex_provenance(self):
        critic = {"localizable": False, "reason": "多个错误交织，不能归因到一点"}
        row, task = self.run_locator(critic)

        self.assertEqual(row["critic"], critic)
        self.assertNotIn("branch_id", row)
        self.assertNotIn("bad_token_logprob", row)
        self.assertEqual(row["critic_gate"]["source"], "codex_as_critic")
        self.assertNotIn("model", row["critic_gate"])
        self.assertTrue(
            Path(row["critic_gate"]["critic_results"]).is_absolute()
        )
        prompt = (
            Path(__file__).resolve().parents[1]
            / "prompt/codex/错误分叉定位.md"
        )
        self.assertEqual(row["critic_gate"]["critic_prompt"], str(prompt))
        self.assertEqual(
            row["critic_gate"]["critic_prompt_sha256"],
            hashlib.sha256(prompt.read_bytes()).hexdigest(),
        )
        self.assertEqual(row["critic_task_id"], task["task_id"])

    def test_rejects_incorrect_error_span(self):
        with self.assertRaisesRegex(ValueError, "invalid critic error span"):
            self.run_locator(
                {
                    "localizable": True,
                    "error_span": "错误",
                    "start_char": 2,
                    "end_char": 4,
                    "error_type": "错误事实",
                    "reason": "位置错误",
                }
            )

    def test_requires_complete_high_fail_from_same_model(self):
        critic = {"localizable": False, "reason": "无需定位"}
        invalid_sources = (
            {"behavior_policy": "different-model"},
            {"candidate_overrides": {"correctness_pass": True}},
            {"candidate_overrides": {"likelihood_region": "middle"}},
            {"candidate_overrides": {"token_ids": [10, True]}},
            {"candidate_overrides": {"raw_token_logprobs": [-0.1]}},
            {
                "candidate_overrides": {
                    "raw_token_logprobs": [-0.1] * 5 + [math.inf]
                }
            },
            {"candidate_overrides": {"num_tokens": 5}},
        )
        for invalid in invalid_sources:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.run_locator(critic, **invalid)

    def test_rejects_invalid_packet_provenance(self):
        critic = {"localizable": False, "reason": "无需定位"}
        invalid_packets = (
            {"prompt_token_ids": [101, True]},
            {"engine": "not-an-object"},
            {"correctness_gate": []},
        )
        for packet_overrides in invalid_packets:
            with self.subTest(packet_overrides=packet_overrides):
                with self.assertRaises(ValueError):
                    self.run_locator(
                        critic,
                        packet_overrides=packet_overrides,
                    )

    def test_maps_split_byte_tokens_to_first_token(self):
        row, _ = self.run_locator(
            {
                "localizable": True,
                "error_span": "😀",
                "start_char": 1,
                "end_char": 2,
                "error_type": "错误事实",
                "reason": "表情位置是错误分叉",
            },
            candidate_overrides={
                "response": "前😀后",
                "token_ids": [20, 21, 22, 23, 24],
                "raw_token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
                "raw_cumulative_logprob": -1.5,
                "raw_mean_token_logprob": -0.3,
                "num_tokens": 5,
            },
            tokenizer=ByteSplitTokenizer(),
        )
        self.assertEqual(row["pivot_index"], 1)
        self.assertEqual(row["bad_token_id"], 21)
        self.assertEqual(row["bad_token_logprob"], -0.2)

    def test_rejects_forged_source_judge_hash(self):
        with self.assertRaisesRegex(ValueError, "invalid judge input hash"):
            self.run_locator(
                {"localizable": False, "reason": "无需定位"},
                candidate_overrides={"judge_input_sha256": "f" * 64},
            )

    def test_rejects_result_for_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidates = directory / "candidates.jsonl"
            tasks = directory / "tasks.jsonl"
            results = directory / "results.jsonl"
            output = directory / "output.jsonl"
            packet = self.make_packet()
            candidates.write_text(json.dumps(packet) + "\n")
            self.call_main(["prepare", candidates, tasks, "--model", "model"])
            task = json.loads(tasks.read_text())

            packet["candidates"][1]["correctness_reason"] = "已经换成另一条错误"
            candidates.write_text(json.dumps(packet) + "\n")
            results.write_text(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "localizable": False,
                        "reason": "旧判断",
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(
                ValueError, "invalid or duplicate Codex result"
            ):
                self.call_main(
                    [
                        "finalize",
                        candidates,
                        results,
                        output,
                        "--model",
                        "model",
                    ]
                )

    def test_results_must_be_complete_unique_and_have_no_extras(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidates = directory / "candidates.jsonl"
            tasks = directory / "tasks.jsonl"
            output = directory / "output.jsonl"
            packet = self.make_packet()
            packet["candidates"].append(
                self.make_candidate(packet["messages"], {"sample_index": 8})
            )
            candidates.write_text(json.dumps(packet) + "\n")
            self.call_main(["prepare", candidates, tasks, "--model", "model"])
            task_rows = [
                json.loads(line) for line in tasks.read_text().splitlines()
            ]
            self.assertEqual(len({task["task_id"] for task in task_rows}), 2)
            valid = [
                {
                    "task_id": task["task_id"],
                    "localizable": False,
                    "reason": "不能可靠定位",
                }
                for task in task_rows
            ]
            cases = {
                "missing": valid[:-1],
                "duplicate": [*valid, valid[0]],
                "extra": [
                    *valid,
                    {
                        "task_id": "0" * 64,
                        "localizable": False,
                        "reason": "多余结果",
                    },
                ],
            }
            for label, rows in cases.items():
                with self.subTest(label=label):
                    results = directory / f"{label}.jsonl"
                    results.write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
                    with self.assertRaises(ValueError):
                        self.call_main(
                            [
                                "finalize",
                                candidates,
                                results,
                                output,
                                "--model",
                                "model",
                            ]
                        )

    def test_empty_task_set_is_valid_but_rejects_extra_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidates = directory / "candidates.jsonl"
            tasks = directory / "tasks.jsonl"
            results = directory / "results.jsonl"
            output = directory / "output.jsonl"
            packet = self.make_packet()
            packet["candidates"] = [{"candidate_kind": "high_pass"}]
            candidates.write_text(json.dumps(packet) + "\n")

            self.call_main(["prepare", candidates, tasks, "--model", "model"])
            self.assertEqual(tasks.read_text(), "")
            results.write_text("")
            load_tokenizer = self.call_main(
                ["finalize", candidates, results, output, "--model", "model"]
            )
            self.assertEqual(output.read_text(), "")
            load_tokenizer.assert_not_called()

            results.write_text(
                json.dumps(
                    {
                        "task_id": "0" * 64,
                        "localizable": False,
                        "reason": "多余结果",
                    }
                )
                + "\n"
            )
            with self.assertRaises(ValueError):
                self.call_main(
                    [
                        "finalize",
                        candidates,
                        results,
                        output,
                        "--model",
                        "model",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
