#!/usr/bin/env python3
"""
Runs the agent call of every evaluated model on the operation-substitution attacks
(data/operation_substitution_requests.jsonl, built by benchmark_construction/build_operation_substitution_attacks.py),
with the same prompt, schema and decoding as the main runs.

Outputs: results/agent_substitution_outputs/<model>.jsonl (resumable)
Usage:   python agent_evaluation/run_operation_substitution_evaluation.py --concurrency 2
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lakehouse_governance_benchmark.agent_prompts import (  # noqa: E402
    POLICY_TEXT, SYSTEM_PROMPT, catalog_text, output_schema, user_prompt,
)
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.local_model_client import chat_json  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "agent_substitution_outputs"
MODELS = ["llama3.2:3b", "qwen3:4b", "qwen3:8b", "llama3.1:8b", "gemma3:12b", "qwen3:14b", "phi4:14b",
          "mistral-small3.2:24b", "gpt-oss:20b", "qwen3:32b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    env = build_environment()
    rows = [json.loads(line) for line in open(DATA / "operation_substitution_requests.jsonl")][: args.limit]
    system = SYSTEM_PROMPT.format(catalog=catalog_text(), policy=POLICY_TEXT)
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        path = OUT / (model.replace(":", "_").replace("/", "_") + ".jsonl")
        done = set()
        if path.exists():
            for line in path.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("parsed") is not None:
                    done.add(r["instance_id"])
        todo = [r for r in rows if r["instance_id"] not in done]
        print(f"{model}: {len(todo)} calls to run ({len(done)} already done)", flush=True)
        if not todo:
            continue

        def work(inst):
            out = chat_json(model, system, user_prompt(inst, env), output_schema())
            return {"model": model, "instance_id": inst["instance_id"], "call": "agent", **out}

        t0, n = time.time(), 0
        with ThreadPoolExecutor(args.concurrency) as pool, path.open("a") as f:
            for fut in as_completed([pool.submit(work, r) for r in todo]):
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
                n += 1
                if n % 25 == 0 or n == len(todo):
                    rate = (time.time() - t0) / n
                    print(f"  {model} {n}/{len(todo)} calls, {rate:.2f}s per call, about "
                          f"{rate * (len(todo) - n) / 3600:.1f} h left", flush=True)
    print("SUBSTITUTION_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
