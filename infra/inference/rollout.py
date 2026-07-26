#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out 256 responses per chat with vLLM.")
    parser.add_argument("input", type=Path, help="JSONL with trace_id and messages")
    parser.add_argument("output", type=Path, help="output JSONL")
    parser.add_argument("--model", required=True, help="model ID or local model path")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    rows = []
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.input}:{line_number}: invalid JSON") from error
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("trace_id"), str)
                or not row["trace_id"]
                or not isinstance(row.get("messages"), list)
            ):
                raise ValueError(
                    f"{args.input}:{line_number}: expected non-empty trace_id and messages"
                )
            rows.append(row)

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        trust_remote_code=True,
    )
    sampling_config = {
        "n": 256,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "logprobs": 1,
    }
    sampling = SamplingParams(**sampling_config)

    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            request = llm.chat([row["messages"]], sampling_params=sampling, use_tqdm=False)[0]
            if len(request.outputs) != 256:
                raise RuntimeError(f"vLLM returned {len(request.outputs)} rollouts instead of 256")
            prompt_token_ids = request.prompt_token_ids
            if (
                not isinstance(prompt_token_ids, list)
                or not prompt_token_ids
                or any(type(token_id) is not int for token_id in prompt_token_ids)
            ):
                raise RuntimeError("vLLM did not return valid prompt token IDs")
            rollouts = []
            for sample in sorted(request.outputs, key=lambda item: item.index):
                if sample.logprobs is None:
                    raise RuntimeError("vLLM did not return sampled-token logprobs")
                token_ids = list(sample.token_ids)
                token_logprobs = [
                    candidates[token_id].logprob
                    for token_id, candidates in zip(token_ids, sample.logprobs, strict=True)
                ]
                cumulative = sum(token_logprobs)
                rollouts.append(
                    {
                        "sample_index": sample.index,
                        "response": sample.text,
                        "token_ids": token_ids,
                        "raw_token_logprobs": token_logprobs,
                        "raw_cumulative_logprob": cumulative,
                        "raw_mean_token_logprob": cumulative / len(token_ids) if token_ids else None,
                        "num_tokens": len(token_ids),
                        "finish_reason": sample.finish_reason,
                        "stop_reason": sample.stop_reason,
                    }
                )
            destination.write(
                json.dumps(
                    {
                        "trace_id": row["trace_id"],
                        "messages": row["messages"],
                        "behavior_policy": args.model,
                        "prompt_token_ids": prompt_token_ids,
                        "engine": {
                            "vllm_version": getattr(vllm, "__version__", "unknown"),
                            "tensor_parallel_size": args.tensor_parallel_size,
                            "logprobs_mode": "raw_logprobs",
                        },
                        "sampling": sampling_config,
                        "rollouts": rollouts,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            destination.flush()


if __name__ == "__main__":
    main()
