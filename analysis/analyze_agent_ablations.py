#!/usr/bin/env python3
"""
Analysis of run_agent_ablation_variants.py on the fixed 800-instance subset.

For every ablation label, compared with the same model's main run (temperature 0, no reasoning) on the same instances:
  reasoning / sampling   llm_only and validator_gate pipelines computed from the variant's own agent output
  self_check             llm_self_check: the reviewer's final decision applied to the main run's proposed operation,
                         against llm_only and validator_gate of the main run
Reported: unsafe execution, decision accuracy, correct automation recall, with bootstrap intervals; McNemar tests of
unsafe execution against the main run (validator_gate for self_check and llm_only otherwise); for sampling, the
share of instances whose decision changes between seeds.

Outputs (results/agent_ablation_analysis/): ablation_metrics.csv, ablation_tests.csv, sampling_stability.csv,
ablation_summary.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent_evaluation"))
sys.path.insert(0, str(ROOT / "analysis"))

from analyze_agent_governance_results import boot_ci, mcnemar, holm, stratum  # noqa: E402
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.pipelines import (  # noqa: E402
    llm_decision, normalize_operation, outcome, validator_decision,
)
from run_local_model_agent_evaluation import load_instances, model_file  # noqa: E402

ABL = ROOT / "results" / "agent_ablation_outputs"
OUT = ROOT / "results" / "agent_ablation_analysis"


def read(path, call):
    out = {}
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["call"] == call and (r.get("parsed") is not None or r["instance_id"] not in out):
                out[r["instance_id"]] = r
    return out


def records(system, pipeline, ids, rows, env, agent, decide):
    recs = []
    for iid in ids:
        if iid not in agent:
            continue
        inst = rows[iid]
        op = normalize_operation(agent[iid].get("parsed"), inst["context"]["requester"])
        d = decide(iid, op, inst)
        recs.append({"system": system, "pipeline": pipeline, "instance_id": iid, "stratum": stratum(inst),
                     "gold_decision": inst["gold_decision"], **outcome(inst, d, op)})
    return recs


def metrics(df):
    rows = []
    for (s, p), g in df.groupby(["system", "pipeline"]):
        auto = g[g.gold_decision == "auto_execute"]
        rows.append({"system": s, "pipeline": p, "n": len(g),
                     "unsafe_execution_rate": g.unsafe_execution.mean(), "unsafe_ci95": boot_ci(g.unsafe_execution),
                     "decision_accuracy": g.decision_correct.mean(),
                     "correct_automation_recall": ((auto.decision == "auto_execute") & auto.operation_correct).mean()
                     if len(auto) else np.nan})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = build_environment()
    rows = load_instances()
    ids = json.load(open(ABL / "subset_instance_ids.json"))
    recs, tests, stability = [], [], []
    labels = sorted(p.stem for p in ABL.glob("*.jsonl"))
    by_model = defaultdict(list)
    for lab in labels:
        sample = next(iter(read(ABL / f"{lab}.jsonl", "agent").values()), None) or \
            next(iter(read(ABL / f"{lab}.jsonl", "self_check").values()), None)
        if sample:
            by_model[sample["model"]].append((lab, sample["variant"]))

    for model, labs in by_model.items():
        base = read(model_file(model), "agent")
        base_name = f"{model} (main run)"
        vdec = lambda iid, op, inst: validator_decision(op, inst["context"], env)  # noqa: E731
        base_llm = records(base_name, "llm_only", ids, rows, env, base, lambda iid, op, inst: llm_decision(base[iid].get("parsed")))
        base_val = records(base_name, "validator_gate", ids, rows, env, base, vdec)
        recs += base_llm + base_val
        for lab, variant in labs:
            name = f"{model} ({variant}{'' if not lab.endswith(('seed1', 'seed2')) else ', ' + lab[-5:]})"
            if variant == "self_check":
                chk = read(ABL / f"{lab}.jsonl", "self_check")
                r = records(name, "llm_self_check", [i for i in ids if i in chk], rows, env, base,
                            lambda iid, op, inst: llm_decision(chk[iid].get("parsed")))
                ref = pd.DataFrame(base_val).set_index("instance_id").unsafe_execution
            else:
                ag = read(ABL / f"{lab}.jsonl", "agent")
                r = records(name, "llm_only", ids, rows, env, ag, lambda iid, op, inst: llm_decision(ag[iid].get("parsed")))
                r += records(name, "validator_gate", ids, rows, env, ag, vdec)
                ref = pd.DataFrame(base_llm).set_index("instance_id").unsafe_execution
                if variant == "sampling":
                    b = {x["instance_id"]: x["decision"] for x in base_llm}
                    v = {x["instance_id"]: x["decision"] for x in r if x["pipeline"] == "llm_only"}
                    common = [i for i in v if i in b]
                    stability.append({"model": model, "label": lab, "instances": len(common),
                                      "llm_decision_changed_vs_temperature0": float(np.mean([v[i] != b[i] for i in common]))})
            recs += r
            first = pd.DataFrame([x for x in r if x["pipeline"] in ("llm_only", "llm_self_check")]) \
                .set_index("instance_id").unsafe_execution
            common = first.index.intersection(ref.index)
            p, n01, n10 = mcnemar(first.loc[common].values, ref.loc[common].values)
            tests.append({"system": name, "pipeline": "llm_self_check" if variant == "self_check" else "llm_only",
                          "compared_with": "main run validator_gate" if variant == "self_check" else "main run llm_only",
                          "n": len(common), "unsafe_variant": float(first.loc[common].mean()),
                          "unsafe_reference": float(ref.loc[common].mean()), "mcnemar_p": p,
                          "variant_only_unsafe": n01, "reference_only_unsafe": n10})
    df = pd.DataFrame(recs).drop_duplicates(subset=["system", "pipeline", "instance_id"])
    m = metrics(df)
    m.to_csv(OUT / "ablation_metrics.csv", index=False)
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t.mcnemar_p.values)
    t.to_csv(OUT / "ablation_tests.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "sampling_stability.csv", index=False)
    summary = {"subset_instances": len(ids), "metrics": m.round(4).to_dict("records"),
               "tests": t.round(4).to_dict("records"), "sampling_stability": stability}
    with open(OUT / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(m.round(4).to_string())
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
