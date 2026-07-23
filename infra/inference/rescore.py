#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore Swift DPO pairs with vLLM.")
    parser.add_argument("input", type=Path, help="Swift DPO JSONL with trace_id")
    parser.add_argument("output", type=Path, help="output JSONL")
    parser.add_argument("--model", required=True, help="model ID or local model path")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.input}:{line_number}: invalid JSON") from error
            messages = row.get("messages") if isinstance(row, dict) else None
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("trace_id"), str)
                or not isinstance(messages, list)
                or not messages
                or not isinstance(messages[-1], dict)
                or messages[-1].get("role") != "assistant"
                or not isinstance(messages[-1].get("content"), str)
                or not isinstance(row.get("rejected_response"), str)
            ):
                raise ValueError(
                    f"{args.input}:{line_number}: expected Swift DPO messages and rejected_response"
                )
            rows.append(row)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)

    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            messages = row["messages"]
            context = messages[:-1]
            prefix_ids = tokenizer.apply_chat_template(
                context,
                tokenize=True,
                add_generation_prompt=True,
            )
            empty_ids = tokenizer.apply_chat_template(
                context + [{"role": "assistant", "content": ""}],
                tokenize=True,
                add_generation_prompt=False,
            )
            if empty_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError("chat template does not preserve the assistant prefix")
            suffix_ids = empty_ids[len(prefix_ids) :]

            prompts = []
            response_ids_list = []
            for response in (messages[-1]["content"], row["rejected_response"]):
                full_ids = tokenizer.apply_chat_template(
                    context + [{"role": "assistant", "content": response}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                response_and_end = full_ids[len(prefix_ids) :]
                if full_ids[: len(prefix_ids)] != prefix_ids or (
                    suffix_ids and response_and_end[-len(suffix_ids) :] != suffix_ids
                ):
                    raise ValueError("chat template does not expose a stable response boundary")
                response_ids = (
                    response_and_end[: -len(suffix_ids)] if suffix_ids else response_and_end
                )
                if not response_ids:
                    raise ValueError("DPO responses must not be empty")
                prompts.append(TokensPrompt(prompt_token_ids=prefix_ids + response_ids))
                response_ids_list.append(response_ids)

            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            scores = []
            for output, response_ids in zip(outputs, response_ids_list, strict=True):
                if output.prompt_logprobs is None:
                    raise RuntimeError("vLLM did not return prompt logprobs")
                positions = output.prompt_logprobs[-len(response_ids) :]
                token_logprobs = [
                    candidates[token_id].logprob
                    for token_id, candidates in zip(response_ids, positions, strict=True)
                    if candidates is not None
                ]
                if len(token_logprobs) != len(response_ids):
                    raise RuntimeError("vLLM omitted a response-token logprob")
                cumulative = sum(token_logprobs)
                scores.append(
                    {
                        "token_ids": response_ids,
                        "raw_token_logprobs": token_logprobs,
                        "raw_cumulative_logprob": cumulative,
                        "raw_mean_token_logprob": cumulative / len(response_ids),
                        "num_tokens": len(response_ids),
                    }
                )

            result = dict(row)
            result["policy_likelihood"] = {
                "model": args.model,
                "chosen": scores[0],
                "rejected": scores[1],
            }
            destination.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            destination.flush()


if __name__ == "__main__":
    main()
