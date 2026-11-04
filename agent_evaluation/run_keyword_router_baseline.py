#!/usr/bin/env python3
"""
Keyword-router baseline: keyword intent rules plus deterministic parameter extraction, with every proposed
operation passed to the codified validator. No model, no training. Writes results/agent_outputs/keyword_router.jsonl
in the same format as the local-model runs.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lakehouse_governance_benchmark.deterministic_parser import extract_parameters, keyword_intent  # noqa: E402
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.pipelines import normalize_operation, validator_decision  # noqa: E402
from run_local_model_agent_evaluation import OUT, evaluation_ids, load_instances  # noqa: E402


def main():
    env = build_environment()
    rows = load_instances()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "keyword_router.jsonl", "w") as f:
        for iid in evaluation_ids(rows):
            inst = rows[iid]
            intent = keyword_intent(inst["text"])
            params = extract_parameters(inst["text"], intent, inst["context"]["requester"], env)
            op = normalize_operation({"intent": intent, "params": params}, inst["context"]["requester"])
            parsed = {"intent": intent, "params": params, "decision": validator_decision(op, inst["context"], env),
                      "confidence": 1.0, "rationale": "keyword rules"}
            f.write(json.dumps({"model": "keyword_router", "instance_id": iid, "call": "agent", "parsed": parsed}) + "\n")
    print("keyword router outputs written")


if __name__ == "__main__":
    main()
