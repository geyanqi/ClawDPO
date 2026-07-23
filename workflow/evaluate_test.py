#!/usr/bin/env python3
import argparse
import json
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


def load_test_set(path: Path) -> dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        trace_id = row.get("trace_id")
        messages = row.get("messages")
        if (
            not isinstance(trace_id, str)
            or trace_id in rows
            or not isinstance(messages, list)
            or not messages
        ):
            raise ValueError(f"{path}: invalid or duplicate test row")
        rows[trace_id] = row
    if not rows:
        raise ValueError("test set is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Test Set replies and machine-judged promotion results."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer")
    infer.add_argument("test_set", type=Path)
    infer.add_argument("output", type=Path)
    infer.add_argument("--model", required=True)
    infer.add_argument("--tensor-parallel-size", type=int, default=8)
    infer.add_argument("--max-tokens", type=int, default=8192)

    judge = commands.add_parser("judge")
    judge.add_argument("test_set", type=Path)
    judge.add_argument("best_responses", type=Path)
    judge.add_argument("candidate_responses", type=Path)
    judge.add_argument("output", type=Path)
    judge.add_argument("--factuality-template", type=Path, required=True)
    judge.add_argument("--quality-template", type=Path, required=True)
    judge.add_argument("--workers", type=int, default=16)
    judge.add_argument("--curl", type=Path)

    args = parser.parse_args()
    test_rows = load_test_set(args.test_set)

    if args.command == "infer":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            generation_config="vllm",
            trust_remote_code=True,
        )
        sampling = SamplingParams(
            n=1,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            seed=0,
        )
        outputs = llm.chat(
            [row["messages"] for row in test_rows.values()],
            sampling_params=sampling,
            use_tqdm=True,
        )
        if len(outputs) != len(test_rows):
            raise RuntimeError("vLLM returned an incomplete Test Set")
        with args.output.open("w", encoding="utf-8") as destination:
            for (trace_id, row), output in zip(
                test_rows.items(), outputs, strict=True
            ):
                if len(output.outputs) != 1:
                    raise RuntimeError(f"{trace_id}: expected exactly one response")
                destination.write(
                    json.dumps(
                        {
                            "trace_id": trace_id,
                            "messages": row["messages"],
                            "model": args.model,
                            "response": output.outputs[0].text,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return

    if args.workers < 1:
        raise ValueError("workers must be positive")

    response_sets = {}
    for label, path in (
        ("best", args.best_responses),
        ("candidate", args.candidate_responses),
    ):
        responses = {}
        for row in read_jsonl(path):
            trace_id = row.get("trace_id")
            if (
                not isinstance(trace_id, str)
                or trace_id in responses
                or trace_id not in test_rows
                or row.get("messages") != test_rows[trace_id]["messages"]
                or not isinstance(row.get("model"), str)
                or not isinstance(row.get("response"), str)
            ):
                raise ValueError(f"{path}: invalid or duplicate response")
            responses[trace_id] = row
        if responses.keys() != test_rows.keys():
            raise ValueError(f"{path}: responses must cover the complete Test Set")
        response_sets[label] = responses

    factuality_template = json.loads(
        args.factuality_template.read_text(encoding="utf-8")
    )
    quality_template = json.loads(args.quality_template.read_text(encoding="utf-8"))
    if not isinstance(factuality_template, dict) or not isinstance(
        quality_template, dict
    ):
        raise ValueError("request templates must be JSON objects")

    root = Path(__file__).resolve().parents[1]
    factuality_prompt = (root / "prompt/事实性检测.md").read_text(encoding="utf-8")
    quality_prompt = (root / "prompt/回复竞对.md").read_text(encoding="utf-8")
    curl = args.curl or root / "infra/cli/curl.sh"

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)

        def evaluate(item: tuple[int, tuple[str, dict]]) -> dict:
            sequence, (trace_id, test_row) = item
            best = response_sets["best"][trace_id]
            candidate = response_sets["candidate"][trace_id]

            def call_judge(
                template: dict, system_prompt: str, payload: dict, suffix: str
            ) -> dict:
                request = dict(template)
                request["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ]
                request_path = temporary / f"{sequence}-{suffix}.json"
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
                response = json.loads(completed.stdout)
                content = response["choices"][0]["message"]["content"]
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("judge response must contain a JSON object")
                return result

            fact_payload = {
                "messages": test_row["messages"],
                "response": best["response"],
            }
            best_fact = call_judge(
                factuality_template,
                factuality_prompt,
                fact_payload,
                "best-factuality",
            )
            fact_payload["response"] = candidate["response"]
            candidate_fact = call_judge(
                factuality_template,
                factuality_prompt,
                fact_payload,
                "candidate-factuality",
            )
            quality = call_judge(
                quality_template,
                quality_prompt,
                {
                    "messages": test_row["messages"],
                    "response_A": best["response"],
                    "response_B": candidate["response"],
                },
                "quality",
            )
            winner = quality.get("winner")
            if (
                not isinstance(best_fact.get("pass"), bool)
                or not isinstance(candidate_fact.get("pass"), bool)
                or winner not in {"A", "B", "tie", "uncertain"}
            ):
                raise ValueError(f"{trace_id}: invalid machine-judge result")
            return {
                "trace_id": trace_id,
                "best_model": best["model"],
                "candidate_model": candidate["model"],
                "best_response": best["response"],
                "candidate_response": candidate["response"],
                "best_factuality_pass": best_fact["pass"],
                "candidate_factuality_pass": candidate_fact["pass"],
                "best_factuality_reason": best_fact.get("reason", ""),
                "candidate_factuality_reason": candidate_fact.get("reason", ""),
                "quality_winner": {"A": "best", "B": "candidate"}.get(
                    winner, winner
                ),
                "quality_reason": quality.get("reason", ""),
            }

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate, enumerate(test_rows.items())))

    with args.output.open("w", encoding="utf-8") as destination:
        for row in results:
            destination.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    print(json.dumps({"test_results": len(results)}))


if __name__ == "__main__":
    main()
