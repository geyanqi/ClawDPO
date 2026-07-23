#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
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


def message_key(messages: object, label: object) -> tuple:
    if not isinstance(messages, list) or not messages or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError(f"{label}: invalid messages")
    return tuple((message["role"], message["content"]) for message in messages)


def contains_test(conversation: tuple, test_conversations: list[tuple]) -> bool:
    return any(
        len(test) <= len(conversation)
        and any(
            conversation[start : start + len(test)] == test
            for start in range(len(conversation) - len(test) + 1)
        )
        for test in test_conversations
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic history and Prompt Pool inputs."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    history = commands.add_parser("history")
    history.add_argument("runs_dir", type=Path)
    history.add_argument("output", type=Path)

    prompts = commands.add_parser("prompts")
    prompts.add_argument("output", type=Path)
    prompts.add_argument("--test-set", type=Path, required=True)
    prompts.add_argument("--pool", type=Path)
    prompts.add_argument("--mined", type=Path, action="append", default=[])

    args = parser.parse_args()

    if args.command == "history":
        collected = []
        for directory in sorted(args.runs_dir.glob("iteration-*")):
            dataset = directory / "dataset.jsonl"
            manifest_path = directory / "manifest.json"
            if not dataset.is_file():
                continue
            if not manifest_path.is_file():
                raise ValueError(f"{directory}: dataset has no manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"{manifest_path}: expected a JSON object")
            for row in read_jsonl(dataset):
                messages = row.get("messages")
                message_key(messages, dataset)
                if (
                    not isinstance(row.get("trace_id"), str)
                    or messages[-1]["role"] != "assistant"
                    or not isinstance(row.get("rejected_response"), str)
                ):
                    raise ValueError(f"{dataset}: invalid Swift DPO row")
                historical = dict(row)
                historical["behavior_policy"] = manifest.get("behavior_policy")
                historical["source_iteration"] = directory.name
                collected.append(historical)

        with args.output.open("w", encoding="utf-8") as destination:
            for row in collected:
                destination.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        print(json.dumps({"chosen_history": len(collected)}))
        return

    test_conversations = [
        message_key(row.get("messages"), args.test_set)
        for row in read_jsonl(args.test_set)
    ]
    if not test_conversations:
        raise ValueError("test set is empty")

    output_rows = []
    seen = set()
    excluded_test = 0
    duplicates = 0
    no_target = 0
    if args.pool:
        for row in read_jsonl(args.pool):
            if not isinstance(row.get("trace_id"), str):
                raise ValueError(f"{args.pool}: prompt row needs trace_id")
            key = message_key(row.get("messages"), args.pool)
            if contains_test(key, test_conversations):
                excluded_test += 1
            elif key in seen:
                duplicates += 1
            else:
                seen.add(key)
                output_rows.append(row)

    for mined_path in args.mined:
        with mined_path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "conversation_detail" not in reader.fieldnames:
                raise ValueError(f"{mined_path}: missing conversation_detail column")
            for line_number, row in enumerate(reader, 2):
                try:
                    detail = json.loads(row["conversation_detail"])
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{mined_path}:{line_number}: invalid conversation_detail"
                    ) from error
                messages = detail.get("messages") if isinstance(detail, dict) else detail
                conversation = message_key(messages, f"{mined_path}:{line_number}")
                if contains_test(conversation, test_conversations):
                    excluded_test += 1
                    continue
                target = next(
                    (
                        index
                        for index in range(len(messages) - 1, 0, -1)
                        if messages[index]["role"] == "assistant"
                        and messages[index - 1]["role"] == "user"
                    ),
                    None,
                )
                if target is None:
                    no_target += 1
                    continue
                context = [
                    {"role": message["role"], "content": message["content"]}
                    for message in messages[:target]
                ]
                key = message_key(context, f"{mined_path}:{line_number}")
                if contains_test(key, test_conversations):
                    excluded_test += 1
                elif key in seen:
                    duplicates += 1
                else:
                    canonical = json.dumps(
                        context,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    seen.add(key)
                    output_rows.append(
                        {
                            "trace_id": "mined-"
                            + hashlib.sha256(canonical.encode()).hexdigest()[:20],
                            "messages": context,
                        }
                    )

    with args.output.open("w", encoding="utf-8") as destination:
        for row in output_rows:
            destination.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    print(
        json.dumps(
            {
                "prompt_pool": len(output_rows),
                "excluded_test": excluded_test,
                "duplicates": duplicates,
                "no_target": no_target,
            }
        )
    )


if __name__ == "__main__":
    main()
