import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from infra.inference import rollout


class FakeSample:
    def __init__(self, index: int):
        self.index = index
        self.text = f"response-{index}"
        self.token_ids = [index + 1000]
        self.logprobs = [
            {index + 1000: SimpleNamespace(logprob=-0.5)}
        ]
        self.finish_reason = "stop"
        self.stop_reason = None


class FakeLLM:
    prompt_token_ids = [9182, 7, 404]
    sampling_params = None

    def __init__(self, **_):
        pass

    def chat(self, *_args, **kwargs):
        self.sampling_params = kwargs["sampling_params"]
        type(self).sampling_params = self.sampling_params
        return [
            SimpleNamespace(
                prompt_token_ids=self.prompt_token_ids,
                outputs=[FakeSample(index) for index in range(256)],
            )
        ]

    def get_tokenizer(self):
        raise AssertionError("rollout must use RequestOutput.prompt_token_ids")


class RolloutTest(unittest.TestCase):
    def run_rollout(self, prompt_token_ids, trace_id="trace") -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "input.jsonl"
            output_path = directory / "output.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "trace_id": trace_id,
                        "messages": [{"role": "user", "content": "question"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            FakeLLM.prompt_token_ids = prompt_token_ids
            fake_vllm = SimpleNamespace(
                __version__="0.test",
                LLM=FakeLLM,
                SamplingParams=lambda **kwargs: kwargs,
            )
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch.object(
                    sys,
                    "argv",
                    [
                        "rollout.py",
                        str(input_path),
                        str(output_path),
                        "--model",
                        "model",
                    ],
                ),
            ):
                rollout.main()
            return json.loads(output_path.read_text(encoding="utf-8"))

    def test_saves_prompt_ids_returned_by_vllm(self) -> None:
        row = self.run_rollout([9182, 7, 404])
        self.assertEqual(row["prompt_token_ids"], [9182, 7, 404])
        self.assertEqual(
            row["engine"],
            {
                "vllm_version": "0.test",
                "tensor_parallel_size": 8,
                "logprobs_mode": "raw_logprobs",
            },
        )
        self.assertEqual(row["sampling"]["n"], 256)
        self.assertEqual(row["sampling"]["seed"], 0)
        self.assertEqual(row["sampling"]["max_tokens"], 8192)
        self.assertEqual(FakeLLM.sampling_params["seed"], 0)

    def test_rejects_invalid_prompt_ids(self) -> None:
        for prompt_token_ids in (None, [], (1, 2), [1, True]):
            with (
                self.subTest(prompt_token_ids=prompt_token_ids),
                self.assertRaisesRegex(RuntimeError, "valid prompt token IDs"),
            ):
                self.run_rollout(prompt_token_ids)

    def test_rejects_empty_trace_id_before_loading_vllm(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty trace_id"):
            self.run_rollout([1], trace_id="")


if __name__ == "__main__":
    unittest.main()
