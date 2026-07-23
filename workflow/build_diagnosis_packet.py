#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean


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


def pair_key(row: dict, path: Path) -> tuple[str, str, str]:
    messages = row.get("messages")
    rejected = row.get("rejected_response")
    if (
        not isinstance(row.get("trace_id"), str)
        or not isinstance(messages, list)
        or not messages
        or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != "assistant"
        or not isinstance(messages[-1].get("content"), str)
        or not isinstance(rejected, str)
    ):
        raise ValueError(f"{path}: invalid DPO pair")
    return row["trace_id"], messages[-1]["content"], rejected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fixed evidence packet for a rejected DPO candidate."
    )
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    manifest_path = args.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "rejected":
        raise ValueError("diagnosis requires a rejected candidate")

    paths = {
        "dataset": args.directory / "dataset.jsonl",
        "draft_pairs": args.directory / "draft-pairs.jsonl",
        "base_scores": args.directory / "base-pair-scores.jsonl",
        "candidate_scores": args.directory / "candidate-pair-scores.jsonl",
        "test_results": args.directory / "test-results.jsonl",
        "test_badcases": args.directory / "test-badcases.jsonl",
        "training_log": args.directory / "training.log",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("missing diagnosis artifacts: " + ", ".join(missing))

    dataset = read_jsonl(paths["dataset"])
    draft_rows = read_jsonl(paths["draft_pairs"])
    base_score_rows = read_jsonl(paths["base_scores"])
    candidate_score_rows = read_jsonl(paths["candidate_scores"])
    drafts = {pair_key(row, paths["draft_pairs"]): row for row in draft_rows}
    base_rows = {
        pair_key(row, paths["base_scores"]): row
        for row in base_score_rows
    }
    candidate_rows = {
        pair_key(row, paths["candidate_scores"]): row
        for row in candidate_score_rows
    }
    dataset_keys = [pair_key(row, paths["dataset"]) for row in dataset]
    if (
        not dataset_keys
        or len(set(dataset_keys)) != len(dataset_keys)
        or len(drafts) != len(draft_rows)
        or len(base_rows) != len(base_score_rows)
        or len(candidate_rows) != len(candidate_score_rows)
        or set(dataset_keys) != drafts.keys()
        or set(dataset_keys) != base_rows.keys()
        or set(dataset_keys) != candidate_rows.keys()
    ):
        raise ValueError("dataset, draft and score files must contain the same unique pairs")

    pair_diagnostics = []
    for row, key in zip(dataset, dataset_keys, strict=True):
        scored = {}
        for label, source, expected_model in (
            ("base", base_rows[key], manifest["behavior_policy"]),
            ("candidate", candidate_rows[key], manifest["candidate_model"]),
        ):
            likelihood = source.get("policy_likelihood", {})
            chosen = likelihood.get("chosen", {})
            rejected = likelihood.get("rejected", {})
            if (
                likelihood.get("model") != expected_model
                or not isinstance(chosen.get("raw_mean_token_logprob"), (int, float))
                or not isinstance(rejected.get("raw_mean_token_logprob"), (int, float))
                or not isinstance(chosen.get("num_tokens"), int)
                or not isinstance(rejected.get("num_tokens"), int)
            ):
                raise ValueError(f"{paths[label + '_scores']}: invalid model or score")
            chosen_logprob = chosen["raw_mean_token_logprob"]
            rejected_logprob = rejected["raw_mean_token_logprob"]
            scored[label] = {
                "chosen_raw_mean_logprob": chosen_logprob,
                "rejected_raw_mean_logprob": rejected_logprob,
                "raw_margin": chosen_logprob - rejected_logprob,
                "chosen_tokens": chosen["num_tokens"],
                "rejected_tokens": rejected["num_tokens"],
            }

        draft = drafts[key]
        pair_type = draft.get("pair_type", "unspecified")
        reason = draft.get("reason", "")
        if not isinstance(pair_type, str) or not isinstance(reason, str):
            raise ValueError(f"{paths['draft_pairs']}: invalid pair metadata")
        pair_diagnostics.append(
            {
                "trace_id": key[0],
                "pair_type": pair_type,
                "selection_reason": reason,
                "messages": row["messages"],
                "rejected_response": row["rejected_response"],
                "base": scored["base"],
                "candidate": scored["candidate"],
                "delta": {
                    "chosen_logprob": (
                        scored["candidate"]["chosen_raw_mean_logprob"]
                        - scored["base"]["chosen_raw_mean_logprob"]
                    ),
                    "rejected_logprob": (
                        scored["candidate"]["rejected_raw_mean_logprob"]
                        - scored["base"]["rejected_raw_mean_logprob"]
                    ),
                    "raw_margin": (
                        scored["candidate"]["raw_margin"]
                        - scored["base"]["raw_margin"]
                    ),
                },
            }
        )

    groups = {"all": pair_diagnostics}
    for row in pair_diagnostics:
        groups.setdefault(row["pair_type"], []).append(row)
    summaries = {}
    for name, rows in groups.items():
        summaries[name] = {
            "pairs": len(rows),
            "raw_margin_improved": sum(
                row["delta"]["raw_margin"] > 0 for row in rows
            ),
            "chosen_logprob_improved": sum(
                row["delta"]["chosen_logprob"] > 0 for row in rows
            ),
            "chosen_down_but_raw_margin_up": sum(
                row["delta"]["chosen_logprob"] < 0
                and row["delta"]["raw_margin"] > 0
                for row in rows
            ),
            "mean_chosen_logprob_delta": mean(
                row["delta"]["chosen_logprob"] for row in rows
            ),
            "mean_rejected_logprob_delta": mean(
                row["delta"]["rejected_logprob"] for row in rows
            ),
            "mean_raw_margin_delta": mean(
                row["delta"]["raw_margin"] for row in rows
            ),
        }

    suspicious = []
    seen = set()
    ranked = sorted(pair_diagnostics, key=lambda row: row["delta"]["raw_margin"])
    ranked += sorted(
        (
            row
            for row in pair_diagnostics
            if row["delta"]["chosen_logprob"] < 0
            and row["delta"]["raw_margin"] > 0
        ),
        key=lambda row: row["delta"]["chosen_logprob"],
    )
    for row in ranked:
        key = (row["trace_id"], row["messages"][-1]["content"], row["rejected_response"])
        if key not in seen:
            suspicious.append(row)
            seen.add(key)
        if len(suspicious) == 20:
            break

    log_lines = paths["training_log"].read_text(encoding="utf-8", errors="replace").splitlines()
    markers = ("loss", "grad_norm", "rewards/", "logps/", "nan", "inf")
    metric_excerpt = [
        line[-2000:] for line in log_lines if any(marker in line.lower() for marker in markers)
    ][-200:]
    test_badcases = read_jsonl(paths["test_badcases"])
    pair_diagnostics_path = args.directory / "pair-diagnostics.jsonl"
    with pair_diagnostics_path.open("w", encoding="utf-8") as destination:
        for row in pair_diagnostics:
            destination.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    packet = {
        "training_triple": {
            "base_model": manifest["behavior_policy"],
            "dataset_revision": manifest["dataset_revision"],
            "candidate_model": manifest["candidate_model"],
        },
        "evaluation_summary": manifest["evaluation_summary"],
        "pair_summary": summaries.pop("all"),
        "pair_type_summaries": summaries,
        "suspicious_pair_sample": suspicious,
        "test_badcase_count": len(test_badcases),
        "test_badcase_sample": test_badcases[:20],
        "training_metric_excerpt": metric_excerpt,
        "artifacts": {
            name: str(path.resolve()) for name, path in paths.items()
        }
        | {
            "pair_diagnostics": str(pair_diagnostics_path.resolve()),
            "training_recipe": str(
                (Path(__file__).resolve().parents[1] / "infra/cli/dpo.sh").resolve()
            ),
        },
    }
    output = args.directory / "diagnosis-packet.json"
    output.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(packet["pair_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
