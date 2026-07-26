#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    critic_prompt_path = root / "prompt/codex/错误分叉定位.md"
    critic_prompt_sha256 = hashlib.sha256(critic_prompt_path.read_bytes()).hexdigest()

    parser = argparse.ArgumentParser(
        description="Locate candidate token branches in high-likelihood failed responses."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="write one Codex task per candidate")
    prepare.add_argument("candidate_packets", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--model", required=True, help="tokenizer model ID or path")

    finalize = commands.add_parser(
        "finalize", help="validate Codex results and map spans to tokens"
    )
    finalize.add_argument("candidate_packets", type=Path)
    finalize.add_argument("results", type=Path)
    finalize.add_argument("output", type=Path)
    finalize.add_argument("--model", required=True, help="tokenizer model ID or path")
    args = parser.parse_args()

    task_ids = []
    selected = {}
    source_keys = set()
    for packet in read_jsonl(args.candidate_packets):
        trace_id = packet.get("trace_id")
        messages = packet.get("messages")
        prompt_token_ids = packet.get("prompt_token_ids")
        if (
            not isinstance(trace_id, str)
            or not trace_id
            or not isinstance(messages, list)
            or not messages
            or packet.get("behavior_policy") != args.model
            or not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(type(token_id) is not int for token_id in prompt_token_ids)
            or not isinstance(packet.get("engine"), dict)
            or not isinstance(packet.get("sampling"), dict)
            or not isinstance(packet.get("correctness_gate"), dict)
            or not isinstance(packet.get("candidates"), list)
        ):
            raise ValueError(f"{args.candidate_packets}: invalid candidate packet")

        for candidate in packet["candidates"]:
            if not isinstance(candidate, dict):
                raise ValueError(f"{trace_id}: invalid candidate")
            if candidate.get("candidate_kind") != "high_fail":
                continue

            sample_index = candidate.get("sample_index")
            response = candidate.get("response")
            token_ids = candidate.get("token_ids")
            token_logprobs = candidate.get("raw_token_logprobs")
            cumulative_logprob = candidate.get("raw_cumulative_logprob")
            mean_logprob = candidate.get("raw_mean_token_logprob")
            num_tokens = candidate.get("num_tokens")
            reason = candidate.get("correctness_reason")
            key = (trace_id, sample_index)
            if (
                type(sample_index) is not int
                or key in source_keys
                or not isinstance(response, str)
                or not isinstance(token_ids, list)
                or not token_ids
                or any(type(token_id) is not int for token_id in token_ids)
                or not isinstance(token_logprobs, list)
                or len(token_logprobs) != len(token_ids)
                or any(
                    isinstance(logprob, bool)
                    or not isinstance(logprob, (int, float))
                    or not math.isfinite(logprob)
                    for logprob in token_logprobs
                )
                or isinstance(cumulative_logprob, bool)
                or not isinstance(cumulative_logprob, (int, float))
                or not math.isfinite(cumulative_logprob)
                or isinstance(mean_logprob, bool)
                or not isinstance(mean_logprob, (int, float))
                or not math.isfinite(mean_logprob)
                or type(num_tokens) is not int
                or num_tokens != len(token_ids)
                or candidate.get("likelihood_region") != "high"
                or candidate.get("correctness_pass") is not False
                or not isinstance(reason, str)
            ):
                raise ValueError(f"{trace_id}: invalid or duplicate high_fail candidate")

            judge_input = json.dumps(
                {"messages": messages, "response": response},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_hash = hashlib.sha256(judge_input.encode()).hexdigest()
            if candidate.get("judge_input_sha256") != expected_hash:
                raise ValueError(f"{trace_id}:{sample_index}: invalid judge input hash")

            critic_input = {
                "messages": messages,
                "response": response,
                "correctness_reason": reason,
            }
            critic_input_json = json.dumps(
                critic_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            task_binding = json.dumps(
                {
                    "kind": "branch_location",
                    "prompt_sha256": critic_prompt_sha256,
                    "trace_id": trace_id,
                    "sample_index": sample_index,
                    **critic_input,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            task_id = hashlib.sha256(task_binding.encode()).hexdigest()
            task_ids.append(task_id)
            source_keys.add(key)
            selected[task_id] = (
                key,
                packet,
                candidate,
                hashlib.sha256(critic_input_json.encode()).hexdigest(),
            )

    if args.command == "prepare":
        with args.output.open("w", encoding="utf-8") as destination:
            for task_id in task_ids:
                _, packet, candidate, _ = selected[task_id]
                destination.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "messages": packet["messages"],
                            "response": candidate["response"],
                            "correctness_reason": candidate["correctness_reason"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return

    raw_critics = {}
    for result in read_jsonl(args.results):
        task_id = result.get("task_id")
        if (
            not isinstance(task_id, str)
            or task_id not in selected
            or task_id in raw_critics
            or type(result.get("localizable")) is not bool
            or not isinstance(result.get("reason"), str)
            or not result["reason"].strip()
        ):
            raise ValueError(f"{args.results}: invalid or duplicate Codex result")

        expected_fields = {"task_id", "localizable", "reason"}
        if result["localizable"]:
            expected_fields.update(
                {"error_span", "start_char", "end_char", "error_type"}
            )
        if set(result) != expected_fields:
            raise ValueError(f"{task_id}: unexpected Codex result fields")
        raw_critics[task_id] = result

    missing = set(task_ids) - raw_critics.keys()
    if missing:
        raise ValueError(f"missing Codex result: {next(iter(missing))}")

    critics = {}
    for task_id in task_ids:
        result = raw_critics[task_id]
        if not result["localizable"]:
            critics[task_id] = {
                "localizable": False,
                "reason": result["reason"],
            }
            continue

        error_span = result.get("error_span")
        start = result.get("start_char")
        end = result.get("end_char")
        error_type = result.get("error_type")
        key, _, candidate, _ = selected[task_id]
        response = candidate["response"]
        if (
            not isinstance(error_span, str)
            or not error_span
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(response)
            or not isinstance(error_type, str)
            or not error_type
            or response[start:end] != error_span
        ):
            raise ValueError(f"{key}: invalid critic error span")
        critics[task_id] = {
            "localizable": True,
            "error_span": error_span,
            "start_char": start,
            "end_char": end,
            "error_type": error_type,
            "reason": result["reason"],
        }

    tokenizer = None
    if any(critic["localizable"] for critic in critics.values()):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    critic_gate = {
        "source": "codex_as_critic",
        "critic_results": str(args.results.resolve()),
        "critic_results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "critic_prompt": str(critic_prompt_path.resolve()),
        "critic_prompt_sha256": critic_prompt_sha256,
    }
    with args.output.open("w", encoding="utf-8") as destination:
        for task_id in task_ids:
            key, packet, candidate, critic_input_sha256 = selected[task_id]
            trace_id, sample_index = key
            critic = critics[task_id]
            output = {
                "trace_id": trace_id,
                "messages": packet["messages"],
                "behavior_policy": packet["behavior_policy"],
                "prompt_token_ids": packet["prompt_token_ids"],
                "source_engine": packet["engine"],
                "source_sampling": packet["sampling"],
                "source_correctness_gate": packet["correctness_gate"],
                "source_sample_index": sample_index,
                "source_response": candidate["response"],
                "source_response_token_ids": candidate["token_ids"],
                "source_raw_mean_token_logprob": candidate[
                    "raw_mean_token_logprob"
                ],
                "source_likelihood_region": candidate["likelihood_region"],
                "source_correctness_pass": candidate["correctness_pass"],
                "source_correctness_reason": candidate["correctness_reason"],
                "source_judge_input_sha256": candidate["judge_input_sha256"],
                "critic": critic,
                "critic_gate": critic_gate,
                "critic_task_id": task_id,
                "critic_input_sha256": critic_input_sha256,
            }
            if critic["localizable"]:
                token_ids = candidate["token_ids"]
                decoded = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if decoded != candidate["response"]:
                    raise ValueError(f"{key}: token IDs do not decode to source response")

                pivot_index = None
                last_boundary = 0
                for index in range(len(token_ids)):
                    prefix = tokenizer.decode(
                        token_ids[: index + 1],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if not candidate["response"].startswith(prefix):
                        continue
                    if len(prefix) <= critic["start_char"]:
                        last_boundary = index + 1
                    else:
                        pivot_index = last_boundary
                        break
                if pivot_index is None:
                    raise ValueError(f"{key}: error start does not map to a token")
                output.update(
                    {
                        "branch_id": f"{trace_id}:{sample_index}:{pivot_index}",
                        "pivot_index": pivot_index,
                        "bad_token_id": token_ids[pivot_index],
                        "bad_token_logprob": candidate["raw_token_logprobs"][
                            pivot_index
                        ],
                    }
                )
            destination.write(
                json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


if __name__ == "__main__":
    main()
