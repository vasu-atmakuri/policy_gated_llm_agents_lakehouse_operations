#!/usr/bin/env python3
"""
Runs the lakehouse operations agent with local models served by Ollama.

For every model and every instance in the evaluation set (plus the threshold-tuning subset of the development
split and, when present, the paraphrase set) two calls are made:
  agent  operation, parameters, decision and confidence
  risk   independent extraction of the effects the request would have
Outputs are appended to results/agent_outputs/<model>.jsonl and finished calls are skipped on restart.

Usage
  python agent_evaluation/run_local_model_agent_evaluation.py --models qwen3:8b gemma3:12b --concurrency 4
  python agent_evaluation/run_local_model_agent_evaluation.py --models qwen3:8b --limit 20     # smoke test
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
    POLICY_TEXT, RISK_SYSTEM_PROMPT, SYSTEM_PROMPT, catalog_text, output_schema, risk_schema, risk_user_prompt,
    user_prompt,
)
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.local_model_client import chat_json  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "agent_outputs"
DEFAULT_MODELS = ["qwen3:8b", "qwen3:14b", "qwen3:32b", "llama3.1:8b", "gemma3:12b", "phi4:14b",
                  "mistral-small3.2:24b", "gpt-oss:20b"]


def load_instances():
    rows = {}
    for name in ("base_requests", "adversarial_requests", "out_of_scope_requests", "paraphrased_requests"):
        p = DATA / f"{name}.jsonl"
        if p.exists():
            for line in p.open():
                r = json.loads(line)
                rows[r["instance_id"]] = r
    return rows


def evaluation_ids(rows):
    m = json.load(open(DATA / "evaluation_manifest.json"))
    ids = m["evaluation_instance_ids"] + m["threshold_tuning_instance_ids"]
    ids += [i for i, r in rows.items() if r.get("split") == "paraphrased"]
    return list(dict.fromkeys(ids))


def model_file(model):
    return OUT / (model.replace(":", "_").replace("/", "_") + ".jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--calls", nargs="+", default=["agent", "risk"])
    args = ap.parse_args()

    env = build_environment()
    rows = load_instances()
    ids = evaluation_ids(rows)[: args.limit]
    system = SYSTEM_PROMPT.format(catalog=catalog_text(), policy=POLICY_TEXT)
    OUT.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        path = model_file(model)
        done = set()
        if path.exists():
            for line in path.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("parsed") is not None:
                    done.add((r["instance_id"], r["call"]))
        todo = [(i, c) for i in ids for c in args.calls if (i, c) not in done]
        print(f"{model}: {len(todo)} calls to run ({len(done)} already done)", flush=True)
        if not todo:
            continue

        def work(item):
            iid, call = item
            inst = rows[iid]
            if call == "agent":
                out = chat_json(model, system, user_prompt(inst, env), output_schema())
            else:
                out = chat_json(model, RISK_SYSTEM_PROMPT, risk_user_prompt(inst), risk_schema())
            return {"model": model, "instance_id": iid, "call": call, **out}

        t0, n = time.time(), 0
        with ThreadPoolExecutor(args.concurrency) as pool, path.open("a") as f:
            futures = [pool.submit(work, it) for it in todo]
            for fut in as_completed(futures):
                r = fut.result()
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                if n % 50 == 0 or n == len(todo):
                    rate = (time.time() - t0) / n
                    print(f"  {model} {n}/{len(todo)} calls, {rate:.2f}s per call, about "
                          f"{rate * (len(todo) - n) / 3600:.1f} h left", flush=True)
    print("AGENT_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
