#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
    parser = argparse.ArgumentParser(
        description="Apply the factuality gate and select rollout candidates."
    )
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("evaluated", type=Path)
    parser.add_argument("candidates", type=Path)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--request-template", type=Path)
    gate.add_argument("--factuality", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = Path(__file__).resolve().parents[1]
    judge_prompt_path = root / "prompt/事实性检测.md"
    judge_prompt = judge_prompt_path.read_text(encoding="utf-8")
    judge_prompt_sha256 = hashlib.sha256(judge_prompt.encode()).hexdigest()

    rollout_rows = read_jsonl(args.rollouts)
    if not rollout_rows:
        raise ValueError("rollout input is empty")
    tasks = []
    judge_inputs = {}
    seen_trace_ids = set()
    for row in rollout_rows:
        if (
            not isinstance(row.get("trace_id"), str)
            or not row["trace_id"]
            or row["trace_id"] in seen_trace_ids
            or not isinstance(row.get("messages"), list)
            or not isinstance(row.get("prompt_token_ids"), list)
            or not row["prompt_token_ids"]
            or any(type(token_id) is not int for token_id in row["prompt_token_ids"])
            or not isinstance(row.get("engine"), dict)
            or not isinstance(row.get("sampling"), dict)
            or not isinstance(row.get("rollouts"), list)
            or len(row["rollouts"]) != 256
        ):
            raise ValueError(
                "rollout rows need unique trace_id, messages, prompt_token_ids, engine, sampling, and 256 rollouts"
            )
        seen_trace_ids.add(row["trace_id"])
        seen_indices = set()
        for sample in row["rollouts"]:
            index = sample.get("sample_index") if isinstance(sample, dict) else None
            score = sample.get("raw_mean_token_logprob") if isinstance(sample, dict) else None
            token_ids = sample.get("token_ids") if isinstance(sample, dict) else None
            token_logprobs = (
                sample.get("raw_token_logprobs") if isinstance(sample, dict) else None
            )
            if (
                not isinstance(sample, dict)
                or not isinstance(index, int)
                or index in seen_indices
                or not isinstance(sample.get("response"), str)
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
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not isinstance(sample.get("raw_cumulative_logprob"), (int, float))
                or not isinstance(sample.get("num_tokens"), int)
            ):
                raise ValueError(f"{row['trace_id']}: invalid rollout sample")
            seen_indices.add(index)
            judge_input = json.dumps(
                {
                    "messages": row["messages"],
                    "response": sample["response"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            judge_inputs[(row["trace_id"], index)] = hashlib.sha256(
                judge_input.encode()
            ).hexdigest()
            tasks.append((row["trace_id"], row["messages"], sample))

    verdicts = {}
    if args.factuality:
        replay_models = set()
        for verdict in read_jsonl(args.factuality):
            key = (verdict.get("trace_id"), verdict.get("sample_index"))
            reason = verdict.get("reason", "")
            judge_model = verdict.get("judge_model")
            if (
                not isinstance(key[0], str)
                or type(key[1]) is not int
                or not isinstance(verdict.get("pass"), bool)
                or not isinstance(reason, str)
                or not isinstance(judge_model, str)
                or not judge_model.strip()
                or verdict.get("judge_input_sha256") != judge_inputs.get(key)
                or key in verdicts
            ):
                raise ValueError(
                    f"{args.factuality}: invalid, stale, or duplicate factuality result"
                )
            verdicts[key] = {
                "pass": verdict["pass"],
                "reason": reason,
            }
            replay_models.add(judge_model.strip())
        if len(replay_models) != 1:
            raise ValueError(f"{args.factuality}: expected one judge_model")
        correctness_gate = {
            "source": "provided_factuality",
            "model": replay_models.pop(),
            "factuality": str(args.factuality.resolve()),
            "factuality_sha256": hashlib.sha256(
                args.factuality.read_bytes()
            ).hexdigest(),
            "judge_prompt": str(judge_prompt_path.resolve()),
            "judge_prompt_sha256": judge_prompt_sha256,
        }
    else:
        request_template = json.loads(args.request_template.read_text(encoding="utf-8"))
        if (
            not isinstance(request_template, dict)
            or not isinstance(request_template.get("model"), str)
            or not request_template["model"].strip()
        ):
            raise ValueError("request template must contain a non-empty model")
        curl = root / "infra/cli/curl.sh"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)

            def evaluate(task: tuple) -> tuple:
                sequence, (trace_id, messages, sample) = task
                request = dict(request_template)
                request["messages"] = [
                    {"role": "system", "content": judge_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"messages": messages, "response": sample["response"]},
                            ensure_ascii=False,
                        ),
                    },
                ]
                request_path = temporary / f"{sequence}.json"
                request_path.write_text(
                    json.dumps(request, ensure_ascii=False),
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [curl, request_path],
                    capture_output=True,
                    text=True,
                )
                if completed.returncode:
                    raise RuntimeError(completed.stderr or completed.stdout)
                payload = json.loads(completed.stdout)
                result = json.loads(payload["choices"][0]["message"]["content"])
                reason = result.get("reason", "") if isinstance(result, dict) else None
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("pass"), bool)
                    or not isinstance(reason, str)
                ):
                    raise ValueError("factuality judge did not return pass and reason")
                return (trace_id, sample["sample_index"]), {
                    "pass": result["pass"],
                    "reason": reason,
                }

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                verdicts = dict(pool.map(evaluate, enumerate(tasks)))
        correctness_gate = {
            "source": "machine_judge",
            "model": request_template["model"],
            "request_template": str(args.request_template.resolve()),
            "request_template_sha256": hashlib.sha256(
                args.request_template.read_bytes()
            ).hexdigest(),
            "judge_prompt": str(judge_prompt_path.resolve()),
            "judge_prompt_sha256": judge_prompt_sha256,
        }

    missing = [
        (trace_id, sample["sample_index"])
        for trace_id, _, sample in tasks
        if (trace_id, sample["sample_index"]) not in verdicts
    ]
    if missing:
        raise ValueError(f"missing factuality results, first missing key: {missing[0]}")

    history_by_trace: dict[str, list[dict]] = {}
    if args.history:
        for row in read_jsonl(args.history):
            messages = row.get("messages")
            likelihood = row.get("policy_likelihood")
            if (
                not isinstance(row.get("trace_id"), str)
                or not isinstance(messages, list)
                or not messages
                or not isinstance(messages[-1], dict)
                or messages[-1].get("role") != "assistant"
                or not isinstance(likelihood, dict)
                or not isinstance(likelihood.get("chosen"), dict)
            ):
                raise ValueError(f"{args.history}: invalid rescored chosen history")
            history_by_trace.setdefault(row["trace_id"], []).append(row)

    with (
        args.evaluated.open("w", encoding="utf-8") as evaluated_output,
        args.candidates.open("w", encoding="utf-8") as candidate_output,
    ):
        for row in rollout_rows:
            ranked = sorted(
                row["rollouts"],
                key=lambda sample: (
                    sample["raw_mean_token_logprob"],
                    sample["sample_index"],
                ),
            )
            count = len(ranked)
            cut_05 = math.ceil(count * 0.05)
            cut_25 = math.ceil(count * 0.25)
            cut_75 = math.ceil(count * 0.75)
            scores = [sample["raw_mean_token_logprob"] for sample in ranked]
            thresholds = {
                "p05": scores[cut_05 - 1],
                "p25": scores[cut_25 - 1],
                "p75": scores[cut_75 - 1],
            }

            evaluated_rollouts = []
            for rank, sample in enumerate(ranked):
                if rank < cut_05:
                    region = "extreme_tail"
                elif rank < cut_25:
                    region = "supported_tail"
                elif rank < cut_75:
                    region = "middle"
                else:
                    region = "high"
                verdict = verdicts[(row["trace_id"], sample["sample_index"])]
                evaluated_sample = dict(sample)
                evaluated_sample["likelihood_region"] = region
                evaluated_sample["correctness_pass"] = verdict["pass"]
                evaluated_sample["correctness_reason"] = verdict["reason"]
                evaluated_sample["judge_input_sha256"] = judge_inputs[
                    (row["trace_id"], sample["sample_index"])
                ]
                evaluated_rollouts.append(evaluated_sample)

            evaluated_row = dict(row)
            evaluated_row["rollouts"] = evaluated_rollouts
            evaluated_row["correctness_gate"] = correctness_gate
            evaluated_output.write(
                json.dumps(evaluated_row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

            unique = {}
            for sample in evaluated_rollouts:
                text = sample["response"].strip()
                if text and (
                    text not in unique
                    or sample["raw_mean_token_logprob"]
                    > unique[text]["raw_mean_token_logprob"]
                ):
                    unique[text] = sample

            supported_pass = sorted(
                (
                    sample
                    for sample in unique.values()
                    if sample["likelihood_region"] == "supported_tail"
                    and sample["correctness_pass"]
                ),
                key=lambda sample: sample["raw_mean_token_logprob"],
            )[:16]
            high_fail = sorted(
                (
                    sample
                    for sample in unique.values()
                    if sample["likelihood_region"] == "high"
                    and not sample["correctness_pass"]
                ),
                key=lambda sample: sample["raw_mean_token_logprob"],
                reverse=True,
            )[:8]
            high_pass = sorted(
                (
                    sample
                    for sample in unique.values()
                    if sample["likelihood_region"] == "high"
                    and sample["correctness_pass"]
                ),
                key=lambda sample: sample["raw_mean_token_logprob"],
                reverse=True,
            )[:8]

            candidates = []
            for kind, selected in (
                ("supported_tail_pass", supported_pass),
                ("high_fail", high_fail),
                ("high_pass", high_pass),
            ):
                for sample in selected:
                    candidates.append(
                        {
                            "candidate_kind": kind,
                            "sample_index": sample["sample_index"],
                            "response": sample["response"],
                            "token_ids": sample["token_ids"],
                            "raw_token_logprobs": sample["raw_token_logprobs"],
                            "raw_cumulative_logprob": sample["raw_cumulative_logprob"],
                            "raw_mean_token_logprob": sample["raw_mean_token_logprob"],
                            "num_tokens": sample["num_tokens"],
                            "likelihood_region": sample["likelihood_region"],
                            "correctness_pass": sample["correctness_pass"],
                            "correctness_reason": sample["correctness_reason"],
                            "judge_input_sha256": sample["judge_input_sha256"],
                        }
                    )

            chosen_history = []
            for historical in history_by_trace.get(row["trace_id"], []):
                if historical["messages"][:-1] != row["messages"]:
                    raise ValueError(f"{row['trace_id']}: chosen history has different context")
                chosen_score = historical["policy_likelihood"]["chosen"]
                mean = chosen_score.get("raw_mean_token_logprob")
                if not isinstance(mean, (int, float)) or not math.isfinite(mean):
                    raise ValueError(f"{row['trace_id']}: chosen history has invalid likelihood")
                if mean <= thresholds["p05"]:
                    region = "extreme_tail"
                elif mean <= thresholds["p25"]:
                    region = "supported_tail"
                elif mean <= thresholds["p75"]:
                    region = "middle"
                else:
                    region = "high"
                chosen_history.append(
                    {
                        "response": historical["messages"][-1]["content"],
                        "source_model": historical.get("behavior_policy"),
                        "scored_by": historical["policy_likelihood"].get("model"),
                        "raw_cumulative_logprob": chosen_score.get(
                            "raw_cumulative_logprob"
                        ),
                        "raw_mean_token_logprob": mean,
                        "num_tokens": chosen_score.get("num_tokens"),
                        "likelihood_region": region,
                    }
                )

            packet = {
                "trace_id": row["trace_id"],
                "messages": row["messages"],
                "behavior_policy": row.get("behavior_policy"),
                "prompt_token_ids": row["prompt_token_ids"],
                "engine": row["engine"],
                "sampling": row["sampling"],
                "correctness_gate": correctness_gate,
                "likelihood_distribution": {
                    "count": count,
                    "min": scores[0],
                    **thresholds,
                    "max": scores[-1],
                },
                "candidates": candidates,
                "chosen_history": chosen_history,
            }
            candidate_output.write(
                json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


if __name__ == "__main__":
    main()
