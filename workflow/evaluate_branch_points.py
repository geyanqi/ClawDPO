#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("original_token", "alternative_token")
ROLLOUT_FIELDS = {
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


def iter_jsonl(path: Path):
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
            yield row


def content_id(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_jsonl_row(destination, row: dict) -> None:
    destination.write(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare anonymous Codex reviews and finalize branch evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("tasks", type=Path)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("input", type=Path)
    finalize.add_argument("codex_results", type=Path)
    finalize.add_argument("evaluated_output", type=Path)
    finalize.add_argument("verified_output", type=Path)
    finalize.add_argument("--min-trials", type=int, default=16)
    finalize.add_argument("--min-gain", type=float, default=0.2)
    finalize.add_argument("--min-pass-rate", type=float, default=0.6)

    for command in (prepare, finalize):
        command.add_argument(
            "--codex-prompt",
            type=Path,
            default=root / "prompt/codex/分叉结果评测.md",
        )
        command.add_argument(
            "--factuality-prompt",
            type=Path,
            default=root / "prompt/事实性检测.md",
        )
        command.add_argument(
            "--quality-prompt",
            type=Path,
            default=root / "prompt/回复竞对.md",
        )
    args = parser.parse_args()

    if args.command == "finalize":
        if args.min_trials < 1:
            raise ValueError("min-trials must be positive")
        if not 0 <= args.min_gain <= 1:
            raise ValueError("min-gain must be between 0 and 1")
        if not 0 <= args.min_pass_rate <= 1:
            raise ValueError("min-pass-rate must be between 0 and 1")

    prompt_paths = {
        "codex": args.codex_prompt,
        "factuality": args.factuality_prompt,
        "quality": args.quality_prompt,
    }
    prompt_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in prompt_paths.items()
    }
    prompt_metadata = {
        name: {
            "path": str(path.resolve()),
            "sha256": prompt_hashes[name],
        }
        for name, path in prompt_paths.items()
    }

    results_by_task = {}
    gate_metadata = None
    if args.command == "finalize":
        for result in iter_jsonl(args.codex_results):
            if set(result) != {
                "task_id",
                "groups",
                "group_winner",
                "group_reason",
            }:
                raise ValueError(f"{args.codex_results}: invalid Codex result")
            task_id = result.get("task_id")
            if (
                not isinstance(task_id, str)
                or not isinstance(result.get("groups"), list)
                or len(result["groups"]) != 2
                or not isinstance(result.get("group_winner"), str)
                or not isinstance(result.get("group_reason"), str)
                or task_id in results_by_task
            ):
                raise ValueError(
                    f"{args.codex_results}: invalid or duplicate Codex result"
                )
            results_by_task[task_id] = result

        results_hash = hashlib.sha256()
        with args.codex_results.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                results_hash.update(chunk)
        gate_metadata = {
            "source": "codex_as_critic",
            "prompts": prompt_metadata,
            "results": str(args.codex_results.resolve()),
            "results_sha256": results_hash.hexdigest(),
        }

    tasks_destination = None
    evaluated_destination = None
    verified_destination = None
    if args.command == "prepare":
        tasks_destination = args.tasks.open("w", encoding="utf-8")
    else:
        evaluated_destination = args.evaluated_output.open("w", encoding="utf-8")
        verified_destination = args.verified_output.open("w", encoding="utf-8")

    branch_count = 0
    task_count = 0
    verified_count = 0
    seen_branch_ids = set()
    seen_task_ids = set()
    used_result_ids = set()

    for row in iter_jsonl(args.input):
        branch_count += 1
        branch_id = row.get("branch_id")
        trace_id = row.get("trace_id")
        sample_index = row.get("source_sample_index")
        pivot_index = row.get("pivot_index")
        bad_token_id = row.get("bad_token_id")
        bad_token_logprob = row.get("bad_token_logprob")
        alternative_token_id = row.get("alternative_token_id")
        alternative_token_text = row.get("alternative_token_text")
        alternative_token_logprob = row.get("alternative_token_raw_logprob")
        source_response = row.get("source_response")
        source_reason = row.get("source_correctness_reason")
        source_ids = row.get("source_response_token_ids")
        messages = row.get("messages")
        critic = row.get("critic")
        sampling = row.get("sampling")
        rollouts = row.get("rollouts")

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
        valid_source_ids = (
            isinstance(source_ids, list)
            and bool(source_ids)
            and all(type(token_id) is int and token_id >= 0 for token_id in source_ids)
        )
        valid_sampling = (
            isinstance(sampling, dict)
            and type(sampling.get("trials")) is int
            and sampling["trials"] >= 1
            and type(sampling.get("selection_seed")) is int
            and sampling["selection_seed"] >= 0
            and type(sampling.get("continuation_seed")) is int
            and sampling["continuation_seed"] >= 0
        )
        if (
            not isinstance(branch_id, str)
            or not branch_id
            or branch_id in seen_branch_ids
            or not isinstance(trace_id, str)
            or not trace_id
            or type(sample_index) is not int
            or sample_index < 0
            or type(pivot_index) is not int
            or not valid_source_ids
            or not 0 <= pivot_index < len(source_ids)
            or type(bad_token_id) is not int
            or bad_token_id < 0
            or not isinstance(bad_token_logprob, (int, float))
            or isinstance(bad_token_logprob, bool)
            or not math.isfinite(bad_token_logprob)
            or type(alternative_token_id) is not int
            or alternative_token_id < 0
            or alternative_token_id == bad_token_id
            or not isinstance(alternative_token_text, str)
            or not isinstance(alternative_token_logprob, (int, float))
            or isinstance(alternative_token_logprob, bool)
            or not math.isfinite(alternative_token_logprob)
            or not isinstance(source_response, str)
            or not isinstance(source_reason, str)
            or not valid_messages
            or not isinstance(critic, dict)
            or critic.get("localizable") is not True
            or row.get("source_likelihood_region") != "high"
            or row.get("source_correctness_pass") is not False
            or not isinstance(row.get("behavior_policy"), str)
            or not row["behavior_policy"]
            or not valid_sampling
            or not isinstance(rollouts, list)
        ):
            raise ValueError(f"{branch_id or args.input}: invalid branch evidence")
        if branch_id != f"{trace_id}:{sample_index}:{pivot_index}":
            raise ValueError(f"{branch_id}: branch_id does not match its source")
        if source_ids[pivot_index] != bad_token_id:
            raise ValueError(f"{branch_id}: bad_token_id does not match pivot")

        expected_source_hash = content_id(
            {"messages": messages, "response": source_response}
        )
        expected_critic_hash = content_id(
            {
                "messages": messages,
                "response": source_response,
                "correctness_reason": source_reason,
            }
        )
        if row.get("source_judge_input_sha256") != expected_source_hash:
            raise ValueError(f"{branch_id}: source judge input hash does not match")
        if row.get("critic_input_sha256") != expected_critic_hash:
            raise ValueError(f"{branch_id}: critic input hash does not match")

        seen_branch_ids.add(branch_id)
        by_key = {}
        replicas = {arm: set() for arm in ARMS}
        source_prefix = source_ids[:pivot_index]
        for rollout in rollouts:
            if not isinstance(rollout, dict) or set(rollout) != ROLLOUT_FIELDS:
                raise ValueError(f"{branch_id}: invalid branch rollout")
            arm = rollout.get("arm")
            replica = rollout.get("replica")
            branch_token_id = rollout.get("branch_token_id")
            branch_token_text = rollout.get("branch_token_text")
            branch_token_logprob = rollout.get("branch_token_raw_logprob")
            response_ids = rollout.get("response_token_ids")
            response = rollout.get("response")
            key = (arm, replica)
            valid_response_ids = (
                isinstance(response_ids, list)
                and len(response_ids) > pivot_index
                and all(
                    type(token_id) is int and token_id >= 0
                    for token_id in response_ids
                )
            )
            if (
                arm not in replicas
                or type(replica) is not int
                or replica < 0
                or type(branch_token_id) is not int
                or branch_token_id < 0
                or not isinstance(branch_token_text, str)
                or not isinstance(branch_token_logprob, (int, float))
                or isinstance(branch_token_logprob, bool)
                or not math.isfinite(branch_token_logprob)
                or not valid_response_ids
                or not isinstance(response, str)
                or key in by_key
            ):
                raise ValueError(f"{branch_id}: invalid or duplicate branch rollout")
            if response_ids[:pivot_index] != source_prefix:
                raise ValueError(f"{branch_id}: rollout prefix does not match source")
            if response_ids[pivot_index] != branch_token_id:
                raise ValueError(f"{branch_id}: rollout branch token does not match IDs")
            if arm == "original_token" and branch_token_id != bad_token_id:
                raise ValueError(f"{branch_id}: original arm changed the source token")
            if (
                arm == "original_token"
                and branch_token_logprob != bad_token_logprob
            ):
                raise ValueError(f"{branch_id}: original token logprob does not match")
            if arm == "alternative_token" and (
                branch_token_id != alternative_token_id
                or branch_token_text != alternative_token_text
                or branch_token_logprob != alternative_token_logprob
            ):
                raise ValueError(f"{branch_id}: alternative token metadata does not match")
            by_key[key] = rollout
            replicas[arm].add(replica)

        expected_replicas = set(range(sampling["trials"]))
        if any(replicas[arm] != expected_replicas for arm in ARMS):
            raise ValueError(f"{branch_id}: branch arms need matching consecutive replicas")

        task_id = content_id(
            {
                "branch_evidence": row,
                "prompt_sha256": prompt_hashes,
            }
        )
        group_contexts = {}
        anonymous_groups = []
        for arm in ARMS:
            group_rollouts = [
                by_key[(arm, replica)] for replica in range(sampling["trials"])
            ]
            group_id = content_id(
                {
                    "task_id": task_id,
                    "responses": [
                        {
                            "response_token_ids": rollout["response_token_ids"],
                            "response": rollout["response"],
                        }
                        for rollout in group_rollouts
                    ],
                }
            )
            response_contexts = {}
            anonymous_responses = []
            for replica, rollout in enumerate(group_rollouts):
                response_id = content_id(
                    {
                        "group_id": group_id,
                        "replica": replica,
                        "response_token_ids": rollout["response_token_ids"],
                        "response": rollout["response"],
                    }
                )
                response_contexts[response_id] = {"replica": replica}
                anonymous_responses.append(
                    {
                        "response_id": response_id,
                        "response": rollout["response"],
                    }
                )
            anonymous_responses.sort(key=lambda response: response["response_id"])
            group_contexts[group_id] = {
                "arm": arm,
                "responses": response_contexts,
            }
            anonymous_groups.append(
                {
                    "group_id": group_id,
                    "responses": anonymous_responses,
                }
            )
        anonymous_groups.sort(key=lambda group: group["group_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"{branch_id}: duplicate Codex task")
        seen_task_ids.add(task_id)
        task_count += 1
        task = {
            "task_id": task_id,
            "messages": messages,
            "groups": anonymous_groups,
        }
        task_context = {
            "branch_id": branch_id,
            "groups": group_contexts,
        }

        if args.command == "prepare":
            write_jsonl_row(tasks_destination, task)
            continue

        if task_id not in results_by_task:
            raise ValueError(
                f"{args.codex_results}: incomplete Codex results, "
                f"first missing: {task_id}"
            )
        used_result_ids.add(task_id)
        result = results_by_task[task_id]
        group_results = result.get("groups")
        group_winner = result.get("group_winner")
        group_reason = result.get("group_reason")
        expected_group_ids = set(task_context["groups"])
        if group_winner not in expected_group_ids | {"tie", "uncertain"}:
            raise ValueError(f"{task_id}: invalid group winner")
        winner_arm = group_winner
        if group_winner in expected_group_ids:
            winner_arm = task_context["groups"][group_winner]["arm"]

        verdicts = {}
        seen_group_ids = set()
        for group_result in group_results:
            if (
                not isinstance(group_result, dict)
                or set(group_result) != {"group_id", "responses"}
            ):
                raise ValueError(f"{task_id}: invalid Codex group result")
            group_id = group_result.get("group_id")
            if (
                not isinstance(group_id, str)
                or group_id not in expected_group_ids
                or group_id in seen_group_ids
                or not isinstance(group_result.get("responses"), list)
            ):
                raise ValueError(f"{task_id}: invalid Codex group result")
            seen_group_ids.add(group_id)
            group_context = task_context["groups"][group_id]
            expected_response_ids = set(group_context["responses"])
            seen_response_ids = set()
            for response_result in group_result["responses"]:
                if (
                    not isinstance(response_result, dict)
                    or set(response_result)
                    != {"response_id", "factuality_pass", "task_pass", "reason"}
                ):
                    raise ValueError(f"{task_id}: invalid Codex response result")
                response_id = response_result.get("response_id")
                if (
                    not isinstance(response_id, str)
                    or response_id not in expected_response_ids
                    or response_id in seen_response_ids
                    or not isinstance(response_result.get("factuality_pass"), bool)
                    or not isinstance(response_result.get("task_pass"), bool)
                    or not isinstance(response_result.get("reason"), str)
                ):
                    raise ValueError(f"{task_id}: invalid Codex response result")
                seen_response_ids.add(response_id)
                replica = group_context["responses"][response_id]["replica"]
                if winner_arm in {"tie", "uncertain"}:
                    group_outcome = winner_arm
                elif group_context["arm"] == winner_arm:
                    group_outcome = "win"
                else:
                    group_outcome = "loss"
                factuality_pass = response_result["factuality_pass"]
                task_pass = response_result["task_pass"]
                verdicts[(group_context["arm"], replica)] = {
                    "task_id": task_id,
                    "group_id": group_id,
                    "response_id": response_id,
                    "factuality_pass": factuality_pass,
                    "task_pass": task_pass,
                    "overall_pass": factuality_pass and task_pass,
                    "reason": response_result["reason"],
                    "group_outcome": group_outcome,
                    "group_reason": group_reason,
                }
            if seen_response_ids != expected_response_ids:
                raise ValueError(f"{task_id}: incomplete Codex response results")
        if seen_group_ids != expected_group_ids:
            raise ValueError(f"{task_id}: incomplete Codex group results")

        quality = {
            "winner": winner_arm,
            "reason": group_reason,
        }

        counts = {
            arm: {
                "factuality": {"passes": 0, "trials": 0},
                "task_completion": {"passes": 0, "trials": 0},
                "overall": {"passes": 0, "trials": 0},
            }
            for arm in ARMS
        }
        evaluated_rollouts = []
        overall_by_replica = {}
        for rollout in row["rollouts"]:
            arm = rollout["arm"]
            replica = rollout["replica"]
            verdict = verdicts[(arm, replica)]
            evaluated_rollout = dict(rollout)
            evaluated_rollout["codex_evaluation"] = verdict
            evaluated_rollouts.append(evaluated_rollout)
            overall_by_replica.setdefault(replica, {})[arm] = verdict["overall_pass"]

            for metric, result_field in (
                ("factuality", "factuality_pass"),
                ("task_completion", "task_pass"),
                ("overall", "overall_pass"),
            ):
                metric_count = counts[arm][metric]
                metric_count["trials"] += 1
                metric_count["passes"] += int(verdict[result_field])

        for arm in ARMS:
            for metric_count in counts[arm].values():
                metric_count["rate"] = (
                    metric_count["passes"] / metric_count["trials"]
                )

        paired_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
        for pair in overall_by_replica.values():
            outcome = (
                f"{int(pair['original_token'])}{int(pair['alternative_token'])}"
            )
            paired_counts[outcome] += 1

        original = counts["original_token"]
        alternative = counts["alternative_token"]
        overall_gain = (
            alternative["overall"]["rate"] - original["overall"]["rate"]
        )
        enough_trials = all(
            counts[arm]["overall"]["trials"] >= args.min_trials for arm in ARMS
        )
        factuality_not_lower = (
            alternative["factuality"]["rate"] >= original["factuality"]["rate"]
        )
        alternative_group_wins = quality["winner"] == "alternative_token"
        verified = (
            enough_trials
            and alternative["overall"]["rate"] >= args.min_pass_rate
            and overall_gain >= args.min_gain
            and factuality_not_lower
            and alternative_group_wins
        )
        alternative_token_id = next(
            rollout["branch_token_id"]
            for rollout in row["rollouts"]
            if rollout["arm"] == "alternative_token"
        )
        summary = {
            "original_token": original,
            "alternative_token": alternative,
            "alternative_token_id": alternative_token_id,
            "overall_gain": overall_gain,
            "paired_counts": paired_counts,
            "quality": quality,
            "factuality_not_lower": factuality_not_lower,
            "alternative_group_wins": alternative_group_wins,
            "criteria": {
                "min_trials": args.min_trials,
                "min_gain": args.min_gain,
                "min_pass_rate": args.min_pass_rate,
                "require_non_decreasing_factuality": True,
                "require_alternative_group_winner": True,
            },
            "enough_trials": enough_trials,
            "verified": verified,
        }

        evaluated_row = dict(row)
        evaluated_row["rollouts"] = evaluated_rollouts
        evaluated_row["gate"] = gate_metadata
        evaluated_row["summary"] = summary
        write_jsonl_row(evaluated_destination, evaluated_row)
        if verified:
            verified_row = dict(evaluated_row)
            verified_row.pop("rollouts")
            write_jsonl_row(verified_destination, verified_row)
            verified_count += 1

    for destination in (
        tasks_destination,
        evaluated_destination,
        verified_destination,
    ):
        if destination is not None:
            destination.close()

    if args.command == "finalize":
        extra = sorted(set(results_by_task) - used_result_ids)
        if extra:
            raise ValueError(
                f"{args.codex_results}: invalid Codex result, "
                f"unknown task: {extra[0]}"
            )
    report = {"branches": branch_count, "tasks": task_count}
    if args.command == "finalize":
        report["verified"] = verified_count
    print(json.dumps(report))


if __name__ == "__main__":
    main()
