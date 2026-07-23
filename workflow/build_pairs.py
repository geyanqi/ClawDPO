#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Codex-selected pairs and write a Swift DPO dataset."
    )
    parser.add_argument("candidate_packets", type=Path)
    parser.add_argument("draft_pairs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    loaded = {}
    for name, path in (
        ("packets", args.candidate_packets),
        ("drafts", args.draft_pairs),
    ):
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
        loaded[name] = rows

    if not loaded["packets"] or not loaded["drafts"]:
        raise ValueError("candidate packets and draft pairs must not be empty")

    packets = {}
    for packet in loaded["packets"]:
        trace_id = packet.get("trace_id")
        if (
            not isinstance(trace_id, str)
            or trace_id in packets
            or not isinstance(packet.get("messages"), list)
            or not isinstance(packet.get("candidates"), list)
            or not isinstance(packet.get("chosen_history"), list)
        ):
            raise ValueError("invalid or duplicate candidate packet")
        packets[trace_id] = packet

    seen_pairs = set()
    with args.output.open("w", encoding="utf-8") as destination:
        for draft in loaded["drafts"]:
            trace_id = draft.get("trace_id")
            messages = draft.get("messages")
            rejected = draft.get("rejected_response")
            if (
                not isinstance(trace_id, str)
                or trace_id not in packets
                or not isinstance(messages, list)
                or not messages
                or not isinstance(messages[-1], dict)
                or messages[-1].get("role") != "assistant"
                or not isinstance(messages[-1].get("content"), str)
                or not isinstance(rejected, str)
            ):
                raise ValueError(f"{trace_id}: invalid Swift DPO draft pair")

            packet = packets[trace_id]
            if messages[:-1] != packet["messages"]:
                raise ValueError(f"{trace_id}: pair context does not match its candidate packet")
            chosen = messages[-1]["content"]
            if chosen == rejected:
                raise ValueError(f"{trace_id}: chosen and rejected must differ")

            sources = []
            for candidate in packet["candidates"]:
                source = dict(candidate)
                source["source"] = "rollout"
                sources.append(source)
            for historical in packet["chosen_history"]:
                source = dict(historical)
                source["source"] = "history"
                sources.append(source)

            chosen_sources = [
                source
                for source in sources
                if source.get("response") == chosen
                and source.get("likelihood_region") != "extreme_tail"
                and (
                    source["source"] == "history"
                    or source.get("correctness_pass") is True
                )
            ]
            rejected_sources = [
                source
                for source in sources
                if source.get("response") == rejected
                and source.get("likelihood_region") == "high"
            ]
            if not chosen_sources:
                raise ValueError(f"{trace_id}: chosen is not an eligible candidate")
            if not rejected_sources:
                raise ValueError(f"{trace_id}: rejected is not a high-likelihood candidate")

            pair_key = (trace_id, chosen, rejected)
            if pair_key in seen_pairs:
                raise ValueError(f"{trace_id}: duplicate preference pair")
            seen_pairs.add(pair_key)
            destination.write(
                json.dumps(
                    {
                        "trace_id": trace_id,
                        "messages": messages,
                        "rejected_response": rejected,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
