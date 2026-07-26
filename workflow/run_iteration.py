#!/usr/bin/env python3
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_report import write_report


BRANCH_ARTIFACTS = (
    "candidate-packets.jsonl",
    "branch-localization-tasks.jsonl",
    "branch-localization-results.jsonl",
    "branch-candidates.jsonl",
    "branch-rollouts.jsonl",
    "branch-outcome-tasks.jsonl",
    "branch-outcome-results.jsonl",
    "evaluated-branches.jsonl",
    "verified-branch-points.jsonl",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_branch_artifacts(
    directory: Path, branch_validation: dict, expected_names: tuple[str, ...]
) -> None:
    artifacts = branch_validation.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_names):
        raise ValueError("branch artifacts are incomplete")
    for name in expected_names:
        path = directory / name
        if not path.is_file() or artifacts[name] != file_sha256(path):
            raise ValueError(f"branch artifact missing or changed: {name}")


def load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{directory}: missing manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(directory: Path, manifest: dict) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        write_report(directory.parent)
    except Exception as error:
        print(f"warning: report update failed: {error}", file=sys.stderr)


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


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def message_key(messages: object, label: object) -> tuple:
    if not isinstance(messages, list) or not messages or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError(f"{label}: invalid messages")
    return tuple((message["role"], message["content"]) for message in messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one versioned ClawDPO iteration.")
    stages = parser.add_subparsers(dest="stage", required=True)

    rollout = stages.add_parser("rollout")
    rollout.add_argument("directory", type=Path)
    rollout.add_argument("prompts", type=Path)
    rollout.add_argument("--model", required=True)
    rollout.add_argument("--tensor-parallel-size", type=int, default=8)
    rollout.add_argument("--max-tokens", type=int, default=8192)
    rollout.add_argument("--seed", type=int, default=0)

    select = stages.add_parser("select")
    select.add_argument("directory", type=Path)
    gate = select.add_mutually_exclusive_group(required=True)
    gate.add_argument("--request-template", type=Path)
    gate.add_argument("--factuality", type=Path)
    select.add_argument("--history", type=Path)
    select.add_argument("--workers", type=int, default=16)

    branch_prepare = stages.add_parser("branch-prepare")
    branch_prepare.add_argument("directory", type=Path)

    branch_rollout = stages.add_parser("branch-rollout")
    branch_rollout.add_argument("directory", type=Path)
    branch_rollout.add_argument("localization_results", type=Path)
    branch_rollout.add_argument("--seed", type=int, default=0)
    branch_rollout.add_argument("--tensor-parallel-size", type=int, default=8)
    branch_rollout.add_argument("--max-tokens", type=int, default=8192)

    branch_evaluate = stages.add_parser("branch-evaluate")
    branch_evaluate.add_argument("directory", type=Path)
    branch_evaluate.add_argument("outcome_results", type=Path)
    branch_evaluate.add_argument("--min-trials", type=int, default=16)
    branch_evaluate.add_argument("--min-gain", type=float, default=0.2)
    branch_evaluate.add_argument("--min-pass-rate", type=float, default=0.6)

    freeze = stages.add_parser("freeze")
    freeze.add_argument("directory", type=Path)
    freeze.add_argument("draft_pairs", type=Path)

    train = stages.add_parser("train")
    train.add_argument("directory", type=Path)

    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("directory", type=Path)
    evaluate.add_argument("test_set", type=Path)
    evaluate.add_argument("results", type=Path)

    diagnose = stages.add_parser("diagnose")
    diagnose.add_argument("directory", type=Path)
    diagnose.add_argument("diagnosis", type=Path)

    mine = stages.add_parser("mine")
    mine.add_argument("directory", type=Path)
    mine.add_argument("where_sql", type=Path)
    mine.add_argument("--database-result", type=Path)

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    branch_lock = None
    if args.stage in {"branch-prepare", "branch-rollout", "branch-evaluate"}:
        branch_lock = (args.directory / ".branch-validation.lock").open("a")
        try:
            fcntl.flock(branch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            branch_lock.close()
            raise ValueError("branch validation is already running") from error

    if args.stage == "rollout":
        args.directory.mkdir(parents=True, exist_ok=False)
        prompts = args.directory / "prompts.jsonl"
        shutil.copy2(args.prompts, prompts)
        manifest = {
            "behavior_policy": args.model,
            "status": "rolling_out",
        }
        write_manifest(args.directory, manifest)
        subprocess.run(
            [
                sys.executable,
                root / "infra/inference/rollout.py",
                prompts,
                args.directory / "rollouts.jsonl",
                "--model",
                args.model,
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--max-tokens",
                str(args.max_tokens),
                "--seed",
                str(args.seed),
            ],
            check=True,
        )
        manifest["status"] = "rolled_out"
        write_manifest(args.directory, manifest)
        return

    manifest = load_manifest(args.directory)

    if args.stage == "select":
        if manifest.get("status") not in {"rolled_out", "selecting"}:
            raise ValueError("select requires a completed rollout")
        manifest["status"] = "selecting"
        write_manifest(args.directory, manifest)
        command = [
            sys.executable,
            root / "workflow/select_candidates.py",
            args.directory / "rollouts.jsonl",
            args.directory / "evaluated-rollouts.jsonl",
            args.directory / "candidate-packets.jsonl",
            "--workers",
            str(args.workers),
        ]
        if args.request_template:
            request_template = args.directory / "factuality-request.json"
            if args.request_template.resolve() != request_template.resolve():
                shutil.copy2(args.request_template, request_template)
            command.extend(["--request-template", request_template])
        else:
            factuality = args.directory / "factuality.jsonl"
            if args.factuality.resolve() != factuality.resolve():
                shutil.copy2(args.factuality, factuality)
            command.extend(["--factuality", factuality])
        if args.history:
            history = args.directory / "chosen-history.jsonl"
            if args.history.resolve() != history.resolve():
                shutil.copy2(args.history, history)
            command.extend(["--history", history])
        subprocess.run(command, check=True)
        manifest["status"] = "selected"
        write_manifest(args.directory, manifest)
        return

    if args.stage == "branch-prepare":
        if manifest.get("status") != "selected" or "branch_validation" in manifest:
            raise ValueError("branch-prepare requires newly selected candidates")
        tasks = args.directory / "branch-localization-tasks.jsonl"
        subprocess.run(
            [
                sys.executable,
                root / "workflow/locate_branch_points.py",
                "prepare",
                args.directory / "candidate-packets.jsonl",
                tasks,
                "--model",
                manifest["behavior_policy"],
            ],
            check=True,
        )
        manifest["branch_validation"] = {
            "status": "localization_tasks_ready",
            "localization_tasks": count_jsonl(tasks),
            "artifacts": {
                name: file_sha256(args.directory / name)
                for name in BRANCH_ARTIFACTS[:2]
            },
        }
        write_manifest(args.directory, manifest)
        branch_lock.close()
        return

    if args.stage == "branch-rollout":
        branch_validation = manifest.get("branch_validation")
        if (
            manifest.get("status") != "selected"
            or not isinstance(branch_validation, dict)
            or branch_validation.get("status") != "localization_tasks_ready"
        ):
            raise ValueError("branch-rollout requires localization tasks")
        validate_branch_artifacts(
            args.directory, branch_validation, BRANCH_ARTIFACTS[:2]
        )

        localization_results = args.directory / "branch-localization-results.jsonl"
        if args.localization_results.resolve() != localization_results.resolve():
            shutil.copy2(args.localization_results, localization_results)
        branch_candidates = args.directory / "branch-candidates.jsonl"
        subprocess.run(
            [
                sys.executable,
                root / "workflow/locate_branch_points.py",
                "finalize",
                args.directory / "candidate-packets.jsonl",
                localization_results,
                branch_candidates,
                "--model",
                manifest["behavior_policy"],
            ],
            check=True,
        )

        branch_rollouts = args.directory / "branch-rollouts.jsonl"
        subprocess.run(
            [
                sys.executable,
                root / "infra/inference/branch_rollout.py",
                branch_candidates,
                branch_rollouts,
                "--model",
                manifest["behavior_policy"],
                "--trials",
                "16",
                "--seed",
                str(args.seed),
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--max-tokens",
                str(args.max_tokens),
            ],
            check=True,
        )

        outcome_tasks = args.directory / "branch-outcome-tasks.jsonl"
        subprocess.run(
            [
                sys.executable,
                root / "workflow/evaluate_branch_points.py",
                "prepare",
                branch_rollouts,
                outcome_tasks,
            ],
            check=True,
        )
        localized = sum(
            row.get("critic", {}).get("localizable") is True
            for row in iter_jsonl(branch_candidates)
        )
        branch_validation.update(
            {
                "status": "outcome_tasks_ready",
                "localization_critic": (
                    "Codex-as-Critic/Branch Localization Critic"
                ),
                "localized_branches": localized,
                "trials_per_token": 16,
                "outcome_tasks": count_jsonl(outcome_tasks),
            }
        )
        branch_validation["artifacts"].update(
            {
                name: file_sha256(args.directory / name)
                for name in BRANCH_ARTIFACTS[2:6]
            }
        )
        write_manifest(args.directory, manifest)
        branch_lock.close()
        return

    if args.stage == "branch-evaluate":
        branch_validation = manifest.get("branch_validation")
        if (
            manifest.get("status") != "selected"
            or not isinstance(branch_validation, dict)
            or branch_validation.get("status") != "outcome_tasks_ready"
        ):
            raise ValueError("branch-evaluate requires outcome tasks")
        validate_branch_artifacts(
            args.directory, branch_validation, BRANCH_ARTIFACTS[:6]
        )

        outcome_results = args.directory / "branch-outcome-results.jsonl"
        if args.outcome_results.resolve() != outcome_results.resolve():
            shutil.copy2(args.outcome_results, outcome_results)
        evaluated = args.directory / "evaluated-branches.jsonl"
        verified = args.directory / "verified-branch-points.jsonl"
        subprocess.run(
            [
                sys.executable,
                root / "workflow/evaluate_branch_points.py",
                "finalize",
                args.directory / "branch-rollouts.jsonl",
                outcome_results,
                evaluated,
                verified,
                "--min-trials",
                str(args.min_trials),
                "--min-gain",
                str(args.min_gain),
                "--min-pass-rate",
                str(args.min_pass_rate),
            ],
            check=True,
        )
        branch_validation.update(
            {
                "status": "evaluated",
                "outcome_critic": "Codex-as-Critic/Branch Outcome Critic",
                "evaluated_branches": count_jsonl(evaluated),
                "verified_branches": count_jsonl(verified),
            }
        )
        branch_validation["artifacts"].update(
            {
                name: file_sha256(args.directory / name)
                for name in BRANCH_ARTIFACTS[6:]
            }
        )
        write_manifest(args.directory, manifest)
        branch_lock.close()
        return

    if args.stage == "freeze":
        branch_validation = manifest.get("branch_validation")
        if (
            manifest.get("status") not in {"selected", "freezing"}
            or not isinstance(branch_validation, dict)
            or branch_validation.get("status") != "evaluated"
        ):
            raise ValueError("freeze requires completed branch validation")
        validate_branch_artifacts(
            args.directory, branch_validation, BRANCH_ARTIFACTS
        )
        manifest["status"] = "freezing"
        write_manifest(args.directory, manifest)
        draft = args.directory / "draft-pairs.jsonl"
        if args.draft_pairs.resolve() != draft.resolve():
            shutil.copy2(args.draft_pairs, draft)
        dataset_revision = (args.directory / "dataset.jsonl").resolve()
        subprocess.run(
            [
                sys.executable,
                root / "workflow/build_pairs.py",
                args.directory / "candidate-packets.jsonl",
                draft,
                dataset_revision,
            ],
            check=True,
        )
        manifest.update(
            {
                "status": "frozen",
                "dataset_revision": str(dataset_revision),
            }
        )
        write_manifest(args.directory, manifest)
        return

    if args.stage == "train":
        if manifest.get("status") not in {"frozen", "training"}:
            raise ValueError("train requires a frozen dataset")
        candidate_model = (args.directory / "candidate-model").resolve()
        training_log = (args.directory / "training.log").resolve()
        manifest.update(
            {
                "status": "training",
                "training_log": str(training_log),
            }
        )
        write_manifest(args.directory, manifest)
        environment = os.environ.copy()
        environment["MODEL_PATH"] = manifest["behavior_policy"]
        environment["OUTPUT_DIR"] = str(candidate_model)
        command = [root / "infra/cli/dpo.sh", args.directory / "dataset.jsonl"]
        with training_log.open("w", encoding="utf-8") as destination:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            if process.stdout is None:
                raise RuntimeError("failed to capture training output")
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                destination.write(line)
                destination.flush()
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        if not candidate_model.is_dir():
            raise ValueError(f"{candidate_model}: training produced no model directory")
        manifest.update(
            {
                "status": "trained",
                "candidate_model": str(candidate_model),
            }
        )
        write_manifest(args.directory, manifest)
        return

    if args.stage == "evaluate":
        if manifest.get("status") not in {"trained", "evaluating"}:
            raise ValueError("evaluate requires a trained candidate")
        candidate_model = (args.directory / "candidate-model").resolve()
        if not candidate_model.is_dir():
            raise ValueError(f"{candidate_model}: missing candidate model")
        manifest["status"] = "evaluating"
        write_manifest(args.directory, manifest)

        test_set_path = args.directory / "test-set.jsonl"
        if args.test_set.resolve() != test_set_path.resolve():
            shutil.copy2(args.test_set, test_set_path)
        test_rows = {}
        for row in read_jsonl(test_set_path):
            trace_id = row.get("trace_id")
            if not isinstance(trace_id, str) or trace_id in test_rows:
                raise ValueError(f"{test_set_path}: invalid or duplicate trace_id")
            message_key(row.get("messages"), test_set_path)
            test_rows[trace_id] = row
        if not test_rows:
            raise ValueError("test set is empty")

        results_path = args.directory / "test-results.jsonl"
        if args.results.resolve() != results_path.resolve():
            shutil.copy2(args.results, results_path)
        counts = {
            "total": 0,
            "best_factuality_failures": 0,
            "candidate_factuality_failures": 0,
            "candidate_wins": 0,
            "best_wins": 0,
            "ties": 0,
            "uncertain": 0,
        }
        base_model = manifest["behavior_policy"]
        candidate_model_name = str(candidate_model)
        result_rows = {}
        for row in read_jsonl(results_path):
            trace_id = row.get("trace_id")
            winner = row.get("quality_winner")
            if (
                not isinstance(trace_id, str)
                or trace_id in result_rows
                or row.get("best_model") != base_model
                or row.get("candidate_model") != candidate_model_name
                or not isinstance(row.get("best_factuality_pass"), bool)
                or not isinstance(row.get("candidate_factuality_pass"), bool)
                or winner not in {"candidate", "best", "tie", "uncertain"}
            ):
                raise ValueError(f"{results_path}: invalid or duplicate result")
            result_rows[trace_id] = row
            counts["total"] += 1
            counts["best_factuality_failures"] += not row["best_factuality_pass"]
            counts["candidate_factuality_failures"] += not row[
                "candidate_factuality_pass"
            ]
            if winner == "candidate":
                counts["candidate_wins"] += 1
            elif winner == "best":
                counts["best_wins"] += 1
            elif winner == "tie":
                counts["ties"] += 1
            else:
                counts["uncertain"] += 1
        if not counts["total"]:
            raise ValueError("test results are empty")
        if result_rows.keys() != test_rows.keys():
            raise ValueError("test results must cover the complete Test Set exactly once")

        promoted = (
            counts["candidate_factuality_failures"]
            <= counts["best_factuality_failures"]
            and counts["candidate_wins"] > counts["best_wins"]
        )
        decision = {
            "promoted": promoted,
            "previous_best_model": base_model,
            "candidate_model": str(candidate_model),
            "next_best_model": str(candidate_model) if promoted else base_model,
            "summary": counts,
        }
        (args.directory / "promotion.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (args.directory / "test-badcases.jsonl").open(
            "w", encoding="utf-8"
        ) as destination:
            for trace_id, test_row in test_rows.items():
                result = result_rows[trace_id]
                if (
                    not result["candidate_factuality_pass"]
                    or result["quality_winner"] != "candidate"
                ):
                    destination.write(
                        json.dumps(
                            {
                                "trace_id": trace_id,
                                "messages": test_row["messages"],
                                "evaluation": result,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        manifest.update(
            {
                "status": "accepted" if promoted else "rejected",
                "previous_best_model": base_model,
                "candidate_model": str(candidate_model),
                "next_best_model": decision["next_best_model"],
                "evaluation_summary": counts,
            }
        )
        write_manifest(args.directory, manifest)
        print(json.dumps(decision, ensure_ascii=False))
        return

    if args.stage == "diagnose":
        if manifest.get("status") != "rejected":
            raise ValueError("diagnose requires a rejected candidate")
        if "diagnosis_summary" in manifest:
            raise ValueError("training failure diagnosis is already recorded")
        diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
        next_attempt = diagnosis.get("next_attempt", {})
        expected_actions = {
            "pipeline_bug": "fix_pipeline",
            "data_problem": "rebuild_dataset",
            "training_problem": "retry_training",
            "insufficient_evidence": "collect_evidence",
        }
        if (
            diagnosis.get("verdict") not in expected_actions
            or not isinstance(diagnosis.get("summary"), str)
            or not diagnosis["summary"].strip()
            or any(
                not isinstance(diagnosis.get(field), list)
                or any(not isinstance(item, str) for item in diagnosis[field])
                for field in ("evidence", "data_findings", "training_findings")
            )
            or not isinstance(next_attempt, dict)
            or next_attempt.get("base_model") != manifest.get("behavior_policy")
            or next_attempt.get("action")
            != expected_actions.get(diagnosis.get("verdict"))
            or not isinstance(next_attempt.get("single_change"), str)
            or not next_attempt["single_change"].strip()
        ):
            raise ValueError("invalid training failure diagnosis")
        destination = args.directory / "diagnosis.json"
        if destination.exists() and args.diagnosis.resolve() != destination.resolve():
            raise ValueError("diagnosis.json already exists")
        if args.diagnosis.resolve() != destination.resolve():
            shutil.copy2(args.diagnosis, destination)
        manifest["diagnosis_summary"] = {
            "verdict": diagnosis["verdict"],
            "summary": diagnosis["summary"],
            "next_action": next_attempt["action"],
            "single_change": next_attempt["single_change"],
        }
        write_manifest(args.directory, manifest)
        return

    if manifest.get("status") not in {"accepted", "rejected"}:
        raise ValueError("mine requires completed test evaluation")
    if "mining_summary" in manifest:
        raise ValueError("training data has already been mined for this iteration")

    test_set_path = args.directory / "test-set.jsonl"
    test_conversations = [
        message_key(row.get("messages"), test_set_path)
        for row in read_jsonl(test_set_path)
    ]
    if not test_conversations:
        raise ValueError("test set is empty")

    query_path = args.directory / "retrieval-where.sql"
    if args.where_sql.resolve() != query_path.resolve():
        shutil.copy2(args.where_sql, query_path)
    response_path = args.directory / "database-response.csv"
    if args.database_result:
        if args.database_result.resolve() != response_path.resolve():
            shutil.copy2(args.database_result, response_path)
    else:
        completed = subprocess.run(
            [root / "infra/cli/database.sh", query_path],
            check=True,
            capture_output=True,
            text=True,
        )
        response_path.write_text(completed.stdout, encoding="utf-8")

    with response_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "conversation_detail" not in reader.fieldnames:
            raise ValueError("database result needs a conversation_detail column")
        rows = list(reader)

    kept = []
    excluded = 0
    for index, row in enumerate(rows, 2):
        try:
            detail = json.loads(row["conversation_detail"])
        except json.JSONDecodeError as error:
            raise ValueError(f"{response_path}:{index}: invalid conversation_detail") from error
        messages = detail.get("messages") if isinstance(detail, dict) else detail
        conversation = message_key(messages, f"{response_path}:{index}")
        is_test_row = any(
            len(test) <= len(conversation)
            and any(
                conversation[start : start + len(test)] == test
                for start in range(len(conversation) - len(test) + 1)
            )
            for test in test_conversations
        )
        if is_test_row:
            excluded += 1
        else:
            kept.append(row)

    output_path = args.directory / "mined-train.csv"
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    manifest["mining_summary"] = {
        "retrieved": len(rows),
        "excluded_test_rows": excluded,
        "kept": len(kept),
    }
    write_manifest(args.directory, manifest)
    print(json.dumps(manifest["mining_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
