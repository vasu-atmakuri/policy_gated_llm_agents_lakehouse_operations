#!/usr/bin/env python3
"""
Analysis of the operation-substitution attacks.

Per model: unsafe execution under llm_only and validator_gate, the share of proposals that follow the planted note
(the proposed table is the substitute table), and dual-model agreement with every other model as partner (the
validator decision, escalated when the two models propose different operations). Bootstrap intervals over instances.

Outputs (results/operation_substitution_analysis/): substitution_metrics.csv, substitution_dual_pairs.csv,
substitution_summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from analyze_agent_governance_results import boot_ci  # noqa: E402
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.pipelines import (  # noqa: E402
    escalate, llm_decision, normalize_operation, outcome, same_operation, validator_decision,
)

OUTPUTS = ROOT / "results" / "agent_substitution_outputs"
OUT = ROOT / "results" / "operation_substitution_analysis"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = build_environment()
    rows = {r["instance_id"]: r for r in map(json.loads, open(ROOT / "data" / "operation_substitution_requests.jsonl"))}
    props = {}
    for p in sorted(OUTPUTS.glob("*.jsonl")):
        d = {}
        for line in p.open():
            r = json.loads(line)
            if r.get("parsed") is not None or r["instance_id"] not in d:
                d[r["instance_id"]] = r.get("parsed")
        props[next(iter(map(json.loads, p.open())))["model"]] = d
    recs = []
    ops = {}
    for model, d in props.items():
        ops[model] = {}
        for iid, parsed in d.items():
            inst = rows[iid]
            op = normalize_operation(parsed, inst["context"]["requester"])
            ops[model][iid] = op
            followed = op is not None and op["params"].get("table") == inst["substitution_table"]
            for pipe, dec in (("llm_only", llm_decision(parsed)),
                              ("validator_gate", validator_decision(op, inst["context"], env))):
                recs.append({"system": model, "pipeline": pipe, "instance_id": iid, "followed_note": followed,
                             "gold_decision": inst["gold_decision"], **outcome(inst, dec, op)})
    df = pd.DataFrame(recs)
    rows_out = []
    for (s, p), g in df.groupby(["system", "pipeline"]):
        rows_out.append({"system": s, "pipeline": p, "n": len(g), "unsafe_execution_rate": g.unsafe_execution.mean(),
                         "unsafe_ci95": boot_ci(g.unsafe_execution), "followed_note_rate": g.followed_note.mean(),
                         "followed_note_ci95": boot_ci(g.followed_note)})
    metrics = pd.DataFrame(rows_out)
    metrics.to_csv(OUT / "substitution_metrics.csv", index=False)
    pairs = []
    for a in ops:
        for b in ops:
            if a == b:
                continue
            u = []
            for iid, oa in ops[a].items():
                if iid not in ops[b]:
                    continue
                inst = rows[iid]
                v = validator_decision(oa, inst["context"], env)
                dec = v if same_operation(oa, ops[b][iid]) else escalate(v)
                u.append(outcome(inst, dec, oa)["unsafe_execution"])
            pairs.append({"system": a, "partner": b, "n": len(u), "unsafe_execution_rate": float(np.mean(u)) if u else np.nan})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "substitution_dual_pairs.csv", index=False)
    summary = {"instances": len(rows), "metrics": metrics.round(4).to_dict("records"),
               "dual_best_partner": pairs.sort_values("unsafe_execution_rate").groupby("system").head(1)
               .round(4).to_dict("records")}
    with open(OUT / "substitution_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()
