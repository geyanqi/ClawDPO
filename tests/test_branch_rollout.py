import hashlib
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "infra/inference/branch_rollout.py"
CRITIC_PROMPT_SHA256 = "a" * 64


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeSample:
    def __init__(
        self,
        token_ids,
        logprobs,
        finish_reason="length",
        stop_reason=None,
    ):
        self.token_ids = token_ids
        self.logprobs = [
            {token_id: FakeLogprob(logprob)}
            for token_id, logprob in zip(token_ids, logprobs, strict=True)
        ]
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason


class FakeRequest:
    def __init__(self, outputs):
        self.outputs = outputs


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTokensPrompt:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens, **_):
        visible = [
            token_id
            for token_id in token_ids
            if not skip_special_tokens or token_id != 99
        ]
        return "|".join(str(token_id) for token_id in visible)


class FakeLLM:
    instances = []
    selection_spec = (40, "length", None)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.instances.append(self)

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        call_number = len(self.calls)
        if call_number == 1:
            return [
                FakeRequest([FakeSample([50 + replica], [-0.5 - replica])])
                for replica in range(len(prompts))
            ]
        if call_number == 2:
            token_id, finish_reason, stop_reason = self.selection_spec
            return [
                FakeRequest(
                    [
                        FakeSample(
                            [token_id],
                            [-0.4],
                            finish_reason,
                            stop_reason,
                        )
                    ]
                )
                for _ in prompts
            ]
        return [
            FakeRequest(
                [
                    FakeSample(
                        [20 + prompt.prompt_token_ids[-1] + replica],
                        [-0.6 - replica],
                    )
                ]
            )
            for replica, prompt in enumerate(prompts)
        ]


def set_evidence_hashes(row):
    inputs = {
        "source_judge_input_sha256": {
            "messages": row["messages"],
            "response": row["source_response"],
        },
        "critic_input_sha256": {
            "messages": row["messages"],
            "response": row["source_response"],
            "correctness_reason": row["source_correctness_reason"],
        },
    }
    for field, value in inputs.items():
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        row[field] = hashlib.sha256(canonical.encode()).hexdigest()
    task_binding = json.dumps(
        {
            "kind": "branch_location",
            "prompt_sha256": row["critic_gate"]["critic_prompt_sha256"],
            "trace_id": row["trace_id"],
            "sample_index": row["source_sample_index"],
            "messages": row["messages"],
            "response": row["source_response"],
            "correctness_reason": row["source_correctness_reason"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    row["critic_task_id"] = hashlib.sha256(task_binding.encode()).hexdigest()
    return row


def branch_row():
    return set_evidence_hashes(
        {
            "branch_id": "trace-1:3:2",
            "trace_id": "trace-1",
            "messages": [{"role": "user", "content": "question"}],
            "behavior_policy": "model-a",
            "prompt_token_ids": [100, 101],
            "source_engine": {"vllm_version": "source-vllm"},
            "source_correctness_gate": {"model": "correctness-model"},
            "critic_gate": {
                "source": "codex_as_critic",
                "critic_prompt_sha256": CRITIC_PROMPT_SHA256,
            },
            "source_sample_index": 3,
            "source_response": "10|20|30|31",
            "source_response_token_ids": [10, 20, 30, 31, 99],
            "source_likelihood_region": "high",
            "source_correctness_pass": False,
            "source_correctness_reason": "wrong fact",
            "critic": {
                "localizable": True,
                "error_span": "30",
                "start_char": 6,
                "end_char": 8,
                "error_type": "wrong fact",
                "reason": "first wrong claim",
            },
            "pivot_index": 2,
            "bad_token_id": 30,
            "bad_token_logprob": -0.3,
        }
    )


class BranchRolloutTest(unittest.TestCase):
    def run_script(
        self,
        row,
        *extra_args,
        selection_spec=None,
        include_vllm_version=True,
    ):
        FakeLLM.instances.clear()
        FakeLLM.selection_spec = selection_spec or (40, "length", None)
        fake_vllm = types.ModuleType("vllm")
        if include_vllm_version:
            fake_vllm.__version__ = "test-vllm"
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        fake_inputs = types.ModuleType("vllm.inputs")
        fake_inputs.TokensPrompt = FakeTokensPrompt
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.jsonl"
            output_path = Path(temporary) / "output.jsonl"
            input_path.write_text(json.dumps(row) + "\n")
            argv = [
                str(SCRIPT),
                str(input_path),
                str(output_path),
                "--model",
                "model-a",
                "--trials",
                "2",
                "--seed",
                "7",
                "--max-tokens",
                "8",
                *extra_args,
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(
                    sys.modules,
                    {"vllm": fake_vllm, "vllm.inputs": fake_inputs},
                ),
            ):
                runpy.run_path(str(SCRIPT), run_name="__main__")
            return output_path.read_text(), FakeLLM.instances

    def test_uses_exact_prefix_and_paired_seeds(self):
        output, instances = self.run_script(branch_row())
        result = json.loads(output)
        self.assertEqual(result["prompt_token_ids"], [100, 101])
        self.assertEqual(
            result["engine"],
            {
                "vllm_version": "test-vllm",
                "tensor_parallel_size": 8,
                "logprobs_mode": "raw_logprobs",
            },
        )

        calls = instances[0].calls
        self.assertEqual(instances[0].kwargs["logprobs_mode"], "raw_logprobs")
        self.assertEqual(
            [prompt.prompt_token_ids for prompt in calls[0][0]],
            [[100, 101, 10, 20, 30], [100, 101, 10, 20, 30]],
        )
        self.assertEqual([params.seed for params in calls[0][1]], [9, 10])
        self.assertTrue(all(params.n == 1 for params in calls[0][1]))
        self.assertTrue(
            all(getattr(params, "logprobs", None) is None for params in calls[0][1])
        )
        self.assertEqual(
            [prompt.prompt_token_ids for prompt in calls[1][0]],
            [[100, 101, 10, 20]],
        )
        self.assertEqual([params.seed for params in calls[1][1]], [7])
        self.assertTrue(all(params.max_tokens == 1 for params in calls[1][1]))
        self.assertTrue(all(params.logprobs == 1 for params in calls[1][1]))
        self.assertTrue(
            all(
                params.logit_bias == {30: float("-inf")}
                for params in calls[1][1]
            )
        )
        self.assertEqual(
            [prompt.prompt_token_ids for prompt in calls[2][0]],
            [
                [100, 101, 10, 20, 40],
                [100, 101, 10, 20, 40],
            ],
        )
        self.assertEqual([params.seed for params in calls[2][1]], [9, 10])
        self.assertTrue(
            all(getattr(params, "logit_bias", None) is None for params in calls[2][1])
        )
        self.assertTrue(
            all(getattr(params, "logprobs", None) is None for params in calls[2][1])
        )

    def test_writes_only_the_needed_result_fields(self):
        output, _ = self.run_script(branch_row())
        result = json.loads(output)
        self.assertEqual(result["behavior_policy"], "model-a")
        self.assertEqual(result["alternative_token_id"], 40)
        self.assertEqual(result["alternative_token_text"], "40")
        self.assertEqual(result["alternative_token_raw_logprob"], -0.4)
        self.assertNotIn("model", result)
        self.assertNotIn("response_prefix_token_ids", result)
        self.assertNotIn("intervention", result)
        self.assertEqual(
            set(result["sampling"]),
            {
                "trials",
                "selection_seed",
                "continuation_seed",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "selection_logprobs",
            },
        )
        self.assertEqual(result["sampling"]["selection_seed"], 7)
        self.assertEqual(result["sampling"]["continuation_seed"], 9)

        expected_rollout_fields = {
            "arm",
            "replica",
            "branch_token_id",
            "branch_token_text",
            "branch_token_raw_logprob",
            "response_token_ids",
            "response",
            "finish_reason",
            "stop_reason",
        }
        self.assertEqual(len(result["rollouts"]), 4)
        self.assertTrue(
            all(set(rollout) == expected_rollout_fields for rollout in result["rollouts"])
        )
        bad = {
            rollout["replica"]: rollout
            for rollout in result["rollouts"]
            if rollout["arm"] == "original_token"
        }
        alternative = {
            rollout["replica"]: rollout
            for rollout in result["rollouts"]
            if rollout["arm"] == "alternative_token"
        }
        self.assertEqual(bad[0]["response_token_ids"], [10, 20, 30, 50])
        self.assertEqual(bad[0]["response"], "10|20|30|50")
        self.assertEqual(alternative[0]["response_token_ids"], [10, 20, 40, 60])
        self.assertEqual(alternative[0]["response"], "10|20|40|60")
        self.assertEqual(alternative[1]["response_token_ids"], [10, 20, 40, 61])
        self.assertEqual(alternative[1]["response"], "10|20|40|61")
        self.assertEqual(bad[0]["branch_token_raw_logprob"], -0.3)
        self.assertEqual(alternative[0]["branch_token_raw_logprob"], -0.4)
        self.assertEqual(
            {rollout["branch_token_id"] for rollout in alternative.values()},
            {40},
        )

    def test_skips_non_localizable_records_without_loading_vllm(self):
        row = {
            "trace_id": "trace-1",
            "critic": {"localizable": False, "reason": "global error"},
        }
        output, instances = self.run_script(row)
        self.assertEqual(output, "")
        self.assertEqual(instances, [])

    def test_rejects_an_alternative_token_that_stops_generation(self):
        with self.assertRaisesRegex(RuntimeError, "alternative token stops generation"):
            self.run_script(
                branch_row(),
                selection_spec=(98, "stop", 98),
            )
        self.assertEqual(len(FakeLLM.instances[0].calls), 2)

    def test_rejects_an_excluded_token_returned_by_vllm(self):
        row = branch_row()
        row["source_response_token_ids"][2] = 40
        row["source_response"] = "10|20|40|31"
        row["bad_token_id"] = 40
        row["critic"]["error_span"] = "40"
        set_evidence_hashes(row)
        with self.assertRaisesRegex(RuntimeError, "excluded bad token"):
            self.run_script(row)

    def test_rejects_source_text_that_does_not_match_its_token_ids(self):
        row = branch_row()
        row["source_response"] = "10|20|30|different"
        set_evidence_hashes(row)
        with self.assertRaisesRegex(ValueError, "source token IDs decode differently"):
            self.run_script(row)

    def test_passes_source_metadata_through_without_interpreting_it(self):
        row = branch_row()
        row["source_engine"] = None
        row["source_correctness_gate"] = "legacy-gate"
        output, _ = self.run_script(row)
        result = json.loads(output)
        self.assertIsNone(result["source_engine"])
        self.assertEqual(result["source_correctness_gate"], "legacy-gate")
        self.assertEqual(result["critic_gate"], row["critic_gate"])

    def test_rejects_a_tampered_critic_task_id(self):
        row = branch_row()
        row["critic_task_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "critic_task_id"):
            self.run_script(row)

    def test_rejects_a_pivot_even_when_bound_fields_are_changed_together(self):
        row = branch_row()
        row["pivot_index"] = 1
        row["bad_token_id"] = 20
        row["branch_id"] = "trace-1:3:1"
        with self.assertRaisesRegex(ValueError, "pivot_index"):
            self.run_script(row)

    def test_uses_unknown_when_vllm_has_no_version(self):
        output, _ = self.run_script(branch_row(), include_vllm_version=False)
        self.assertEqual(json.loads(output)["engine"]["vllm_version"], "unknown")

    def test_rejects_misaligned_input(self):
        invalid_rows = {
            "model": {"behavior_policy": "another-model"},
            "empty prompt tokens": {"prompt_token_ids": []},
            "token ids type": {"source_response_token_ids": 7},
            "pivot token": {"bad_token_id": 31},
            "bad pivot logprob": {"bad_token_logprob": float("nan")},
            "branch id": {"branch_id": "wrong"},
            "critic": {"critic": {"localizable": "yes"}},
            "critic source": {"critic_gate": {"source": "direct_model"}},
            "critic prompt hash": {
                "critic_gate": {
                    "source": "codex_as_critic",
                    "critic_prompt_sha256": "not-a-hash",
                }
            },
            "critic task id": {"critic_task_id": "not-a-hash"},
            "critic span": {
                "critic": {
                    "localizable": True,
                    "error_span": "20",
                    "start_char": 6,
                    "end_char": 8,
                    "error_type": "wrong fact",
                    "reason": "first wrong claim",
                }
            },
            "non-high source": {"source_likelihood_region": "supported_tail"},
            "passing source": {"source_correctness_pass": True},
            "missing source hash": {"source_judge_input_sha256": None},
            "bad critic hash": {"critic_input_sha256": "not-a-hash"},
            "wrong source hash": {"source_judge_input_sha256": "c" * 64},
        }
        for label, changes in invalid_rows.items():
            with self.subTest(label=label):
                row = branch_row()
                row.update(changes)
                with self.assertRaises(ValueError):
                    self.run_script(row)


if __name__ == "__main__":
    unittest.main()
