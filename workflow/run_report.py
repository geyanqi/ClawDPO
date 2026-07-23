#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def write_report(runs_dir: Path, output: Path | None = None) -> None:
    triples = []
    rows = []
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path}: expected a JSON object")
        directory = manifest_path.parent
        triples.append(
            [
                directory.name,
                manifest.get("behavior_policy", "-"),
                manifest.get("dataset_revision", "-"),
                manifest.get("candidate_model", "-"),
            ]
        )
        prompts = count_jsonl(directory / "prompts.jsonl")
        packets = count_jsonl(directory / "candidate-packets.jsonl")
        pairs = count_jsonl(directory / "dataset.jsonl")
        used_traces = set()
        dataset = directory / "dataset.jsonl"
        if dataset.is_file():
            with dataset.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row.get("trace_id"), str):
                            used_traces.add(row["trace_id"])
        skipped = (
            max(packets - len(used_traces), 0)
            if packets is not None and pairs is not None
            else None
        )
        evaluation = manifest.get("evaluation_summary", {})
        mining = manifest.get("mining_summary", {})
        diagnosis = manifest.get("diagnosis_summary", {})
        rows.append(
            [
                directory.name,
                manifest.get("status", "-"),
                prompts,
                pairs,
                skipped,
                (
                    f"{evaluation.get('best_factuality_failures', '-')}/"
                    f"{evaluation.get('candidate_factuality_failures', '-')}"
                ),
                (
                    f"{evaluation.get('candidate_wins', '-')}/"
                    f"{evaluation.get('best_wins', '-')}/"
                    f"{evaluation.get('ties', '-')}/"
                    f"{evaluation.get('uncertain', '-')}"
                ),
                mining.get("kept", "-"),
                (
                    f"{diagnosis['verdict']}/{diagnosis['next_action']}"
                    if diagnosis
                    else "-"
                ),
                manifest.get("next_best_model", "-"),
            ]
        )

    lines = [
        "# ClawDPO 运行报告",
        "",
        "> 由 workflow 在每次状态变化时自动更新。",
        "",
        "## Training Triples",
        "",
        "| Iteration | Base Model | High-quality Dataset | Trained Model |",
        "|---|---|---|---|",
    ]
    for row in triples:
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Iteration Status",
            "",
            "| Iteration | Status | Prompts | Pairs | Skipped | Fact B/C | Quality C/B/T/U | Mined | Diagnosis | Next Best |",
            "|---|---|---:|---:|---:|---|---|---:|---|---|",
        ]
    )
    for row in rows:
        cells = [
            "-" if value is None else str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "`Fact B/C` 是 Best/Candidate 事实性失败数；"
            "`Quality C/B/T/U` 是 Candidate/Best/Tie/Uncertain 数量。",
            "",
        ]
    )
    destination = output or runs_dir / "report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a ClawDPO run report.")
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_report(args.runs_dir, args.output)


if __name__ == "__main__":
    main()
