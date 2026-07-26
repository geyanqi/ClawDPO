#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


def generate_one_each(llm, prompts, sampling_params, error_message):
    requests = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    if len(requests) != len(prompts) or any(
        len(request.outputs) != 1 for request in requests
    ):
        raise RuntimeError(error_message)
    return [request.outputs[0] for request in requests]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roll out matched continuations around localized bad tokens."
    )
    parser.add_argument("input", type=Path, help="localized branch-point JSONL")
    parser.add_argument("output", type=Path, help="output JSONL")
    parser.add_argument("--model", required=True, help="model ID or local model path")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    rows = []
    branch_ids = set()
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{args.input}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON") from error

            critic = row.get("critic") if isinstance(row, dict) else None
            if not isinstance(critic, dict) or type(critic.get("localizable")) is not bool:
                raise ValueError(
                    f"{location}: expected critic with boolean localizable"
                )
            if not critic["localizable"]:
                continue

            messages = row.get("messages")
            prompt_ids = row.get("prompt_token_ids")
            response_ids = row.get("source_response_token_ids")
            source_response = row.get("source_response")
            pivot = row.get("pivot_index")
            bad_token = row.get("bad_token_id")
            bad_logprob = row.get("bad_token_logprob")
            critic_gate = row.get("critic_gate")
            critic_prompt_sha256 = (
                critic_gate.get("critic_prompt_sha256")
                if isinstance(critic_gate, dict)
                else None
            )
            critic_task_id = row.get("critic_task_id")
            valid_messages = (
                isinstance(messages, list)
                and bool(messages)
                and all(
                    isinstance(message, dict)
                    and isinstance(message.get("role"), str)
                    and isinstance(message.get("content"), str)
                    for message in messages
                )
            )
            valid_prompt_ids = (
                isinstance(prompt_ids, list)
                and bool(prompt_ids)
                and all(type(token_id) is int and token_id >= 0 for token_id in prompt_ids)
            )
            valid_response_ids = (
                isinstance(response_ids, list)
                and bool(response_ids)
                and all(
                    type(token_id) is int and token_id >= 0
                    for token_id in response_ids
                )
            )
            valid_critic = (
                isinstance(source_response, str)
                and set(critic)
                == {
                    "localizable",
                    "error_span",
                    "start_char",
                    "end_char",
                    "error_type",
                    "reason",
                }
                and isinstance(critic.get("error_span"), str)
                and bool(critic["error_span"])
                and type(critic.get("start_char")) is int
                and type(critic.get("end_char")) is int
                and 0
                <= critic["start_char"]
                < critic["end_char"]
                <= len(source_response)
                and source_response[critic["start_char"] : critic["end_char"]]
                == critic["error_span"]
                and isinstance(critic.get("error_type"), str)
                and bool(critic["error_type"])
                and isinstance(critic.get("reason"), str)
                and bool(critic["reason"].strip())
            )
            evidence_hashes = (
                row.get("source_judge_input_sha256"),
                row.get("critic_input_sha256"),
                critic_prompt_sha256,
                critic_task_id,
            )
            valid_evidence_hashes = all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdefABCDEF" for character in value)
                for value in evidence_hashes
            )
            if (
                not isinstance(row.get("branch_id"), str)
                or not row["branch_id"]
                or not isinstance(row.get("trace_id"), str)
                or not row["trace_id"]
                or not valid_messages
                or row.get("behavior_policy") != args.model
                or type(row.get("source_sample_index")) is not int
                or row["source_sample_index"] < 0
                or row.get("source_likelihood_region") != "high"
                or row.get("source_correctness_pass") is not False
                or not isinstance(row.get("source_correctness_reason"), str)
                or not isinstance(critic_gate, dict)
                or critic_gate.get("source") != "codex_as_critic"
                or not valid_critic
                or not valid_prompt_ids
                or not valid_response_ids
                or not valid_evidence_hashes
                or type(pivot) is not int
                or not 0 <= pivot < len(response_ids)
                or type(bad_token) is not int
                or bad_token < 0
                or not isinstance(bad_logprob, (int, float))
                or isinstance(bad_logprob, bool)
                or not math.isfinite(bad_logprob)
            ):
                raise ValueError(f"{location}: invalid localized branch record")
            if response_ids[pivot] != bad_token:
                raise ValueError(f"{location}: bad_token_id does not match pivot_index")

            evidence_inputs = (
                (
                    "source_judge_input_sha256",
                    {"messages": messages, "response": source_response},
                ),
                (
                    "critic_input_sha256",
                    {
                        "messages": messages,
                        "response": source_response,
                        "correctness_reason": row["source_correctness_reason"],
                    },
                ),
            )
            for field, evidence_input in evidence_inputs:
                canonical = json.dumps(
                    evidence_input,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
                if row[field] != expected_hash:
                    raise ValueError(f"{location}: {field} does not match its input")

            task_binding = json.dumps(
                {
                    "kind": "branch_location",
                    "prompt_sha256": critic_prompt_sha256,
                    "trace_id": row["trace_id"],
                    "sample_index": row["source_sample_index"],
                    "messages": messages,
                    "response": source_response,
                    "correctness_reason": row["source_correctness_reason"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if critic_task_id != hashlib.sha256(task_binding.encode()).hexdigest():
                raise ValueError(f"{location}: critic_task_id does not match its input")

            expected_branch_id = (
                f"{row['trace_id']}:{row['source_sample_index']}:{pivot}"
            )
            if row["branch_id"] != expected_branch_id:
                raise ValueError(f"{location}: branch_id does not match its source")
            if row["branch_id"] in branch_ids:
                raise ValueError(f"{location}: duplicate branch_id {row['branch_id']}")
            branch_ids.add(row["branch_id"])
            rows.append(row)

    if not rows:
        args.output.write_text("", encoding="utf-8")
        return

    import vllm
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
    continuation_seed = args.seed + args.trials
    common_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }

    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            response_ids = row["source_response_token_ids"]
            pivot = row["pivot_index"]
            bad_token = row["bad_token_id"]
            prefix_ids = response_ids[:pivot]
            prompt_ids = row["prompt_token_ids"]
            decoded_source = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if decoded_source != row["source_response"]:
                raise ValueError(f"{row['branch_id']}: source token IDs decode differently")
            mapped_pivot = None
            last_boundary = 0
            for index in range(len(response_ids)):
                decoded_prefix = tokenizer.decode(
                    response_ids[: index + 1],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if not row["source_response"].startswith(decoded_prefix):
                    continue
                if len(decoded_prefix) <= row["critic"]["start_char"]:
                    last_boundary = index + 1
                else:
                    mapped_pivot = last_boundary
                    break
            if mapped_pivot is None:
                raise ValueError(
                    f"{row['branch_id']}: critic error start does not map to a token"
                )
            if mapped_pivot != pivot:
                raise ValueError(
                    f"{row['branch_id']}: pivot_index does not match critic error span"
                )
            if bad_token == tokenizer.eos_token_id:
                raise ValueError(
                    f"{row['branch_id']}: early_stop at the primary EOS is unsupported"
                )

            bad_prompts = [
                TokensPrompt(prompt_token_ids=prompt_ids + prefix_ids + [bad_token])
                for _ in range(args.trials)
            ]
            bad_samples = generate_one_each(
                llm,
                bad_prompts,
                [
                    SamplingParams(
                        n=1,
                        seed=continuation_seed + replica,
                        max_tokens=args.max_tokens,
                        **common_sampling,
                    )
                    for replica in range(args.trials)
                ],
                "vLLM returned incomplete bad-token trials",
            )

            selection = generate_one_each(
                llm,
                [TokensPrompt(prompt_token_ids=prompt_ids + prefix_ids)],
                [
                    SamplingParams(
                        n=1,
                        seed=args.seed,
                        max_tokens=1,
                        logit_bias={bad_token: float("-inf")},
                        logprobs=1,
                        **common_sampling,
                    )
                ],
                "vLLM did not return one alternative token",
            )[0]
            selected_ids = list(selection.token_ids)
            if len(selected_ids) != 1:
                raise RuntimeError("vLLM did not return one alternative token")
            alternative_token = selected_ids[0]
            if alternative_token == bad_token:
                raise RuntimeError("vLLM sampled the excluded bad token")
            if selection.finish_reason == "stop":
                raise RuntimeError("the alternative token stops generation")
            if (
                selection.logprobs is None
                or len(selection.logprobs) != 1
                or alternative_token not in selection.logprobs[0]
            ):
                raise RuntimeError("vLLM did not return the alternative token logprob")
            alternative_logprob = selection.logprobs[0][alternative_token].logprob
            alternative_token_text = tokenizer.decode(
                [alternative_token],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            alternative_prompts = [
                TokensPrompt(
                    prompt_token_ids=prompt_ids
                    + prefix_ids
                    + [alternative_token]
                )
                for _ in range(args.trials)
            ]
            alternative_samples = generate_one_each(
                llm,
                alternative_prompts,
                [
                    SamplingParams(
                        n=1,
                        seed=continuation_seed + replica,
                        max_tokens=args.max_tokens,
                        **common_sampling,
                    )
                    for replica in range(args.trials)
                ],
                "vLLM returned incomplete alternative-token trials",
            )

            rollouts = []
            for replica in range(args.trials):
                bad_sample = bad_samples[replica]
                bad_response_ids = (
                    prefix_ids + [bad_token] + list(bad_sample.token_ids)
                )
                rollouts.append(
                    {
                        "arm": "original_token",
                        "replica": replica,
                        "branch_token_id": bad_token,
                        "branch_token_text": tokenizer.decode(
                            [bad_token],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        "branch_token_raw_logprob": row["bad_token_logprob"],
                        "response_token_ids": bad_response_ids,
                        "response": tokenizer.decode(
                            bad_response_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "finish_reason": bad_sample.finish_reason,
                        "stop_reason": bad_sample.stop_reason,
                    }
                )

                continuation = alternative_samples[replica]
                continuation_ids = list(continuation.token_ids)
                alternative_response_ids = (
                    prefix_ids + [alternative_token] + continuation_ids
                )
                rollouts.append(
                    {
                        "arm": "alternative_token",
                        "replica": replica,
                        "branch_token_id": alternative_token,
                        "branch_token_text": alternative_token_text,
                        "branch_token_raw_logprob": alternative_logprob,
                        "response_token_ids": alternative_response_ids,
                        "response": tokenizer.decode(
                            alternative_response_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "finish_reason": continuation.finish_reason,
                        "stop_reason": continuation.stop_reason,
                    }
                )

            result = dict(row)
            for duplicate in ("response_prefix_token_ids", "model", "intervention"):
                result.pop(duplicate, None)
            result.update(
                {
                    "engine": {
                        "vllm_version": getattr(vllm, "__version__", "unknown"),
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "logprobs_mode": "raw_logprobs",
                    },
                    "sampling": {
                        "trials": args.trials,
                        "selection_seed": args.seed,
                        "continuation_seed": continuation_seed,
                        "max_tokens": args.max_tokens,
                        "selection_logprobs": 1,
                        **common_sampling,
                    },
                    "alternative_token_id": alternative_token,
                    "alternative_token_text": alternative_token_text,
                    "alternative_token_raw_logprob": alternative_logprob,
                    "rollouts": rollouts,
                }
            )
            destination.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            destination.flush()


if __name__ == "__main__":
    main()
