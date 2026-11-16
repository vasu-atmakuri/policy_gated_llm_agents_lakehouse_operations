#!/usr/bin/env python3
"""
Scores the returned targeted-audit sheets and re-reports the residual failures without disputed gold labels.

Input: a folder holding targeted_audit_sheet_rater_a.xlsx and targeted_audit_sheet_rater_b.xlsx as returned. The raw
files are not copied into the repository; only the answer columns are kept, without file metadata, in
data/human_policy_validation/returned_responses/.

A request is disputed when both raters' operations differ from the gold operation (type or any argument), both mark it
ambiguous, or both give a decision different from the gold decision.

Outputs (results/human_validation_analysis/):
  targeted_audit_agreement.csv        agreement with gold per rater and item type (targeted, control)
  targeted_audit_inter_rater.csv      Cohen's kappa between the raters and between each rater and gold
  targeted_audit_items.csv            every item with gold labels, both raters' answers and the dispute flag
  residual_failures_without_disputed.csv  unsafe executions per model and pipeline with and without disputed requests
  targeted_audit_summary.json
Usage: python analysis/score_targeted_audit_sheets.py /path/to/returned_sheets
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from score_human_validation_sheets import ARGS, DECISIONS, arg_match, kappa, read_sheet  # noqa: E402

DATA = ROOT / "data" / "human_policy_validation"
OUT = ROOT / "results" / "human_validation_analysis"
COLS = ["operation", "resource", "principal", "number", "option", "ambiguous", "decision", "notes"]
PIPELINES = {"validator_gate": "validator_gate", "dual_gemma": "dual_model_agreement|gemma3:12b",
             "dual_mistral": "dual_model_agreement|mistral-small3.2:24b"}


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "returned_responses").mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(DATA / "targeted_audit_answer_key.csv", dtype=str, keep_default_na=False)
    key["item_id"] = key.item_id.astype(int)
    t = key.copy()
    for r in ("a", "b"):
        s = read_sheet(folder, f"targeted_audit_sheet_rater_{r}.xlsx", COLS)
        for c in ("operation", "decision", "ambiguous"):
            s[c] = s[c].str.strip().str.lower()
        s.to_csv(DATA / "returned_responses" / f"targeted_audit_responses_rater_{r}.csv", index=False)
        t = t.merge(s.rename(columns={c: f"{c}_{r}" for c in COLS}), on="item_id")
        t[f"op_{r}_ok"] = t[f"operation_{r}"] == t.gold_operation
        full = t[f"op_{r}_ok"].copy()
        for c in ARGS:
            full &= np.array([arg_match(c, rv, gv, req) for rv, gv, req in zip(t[f"{c}_{r}"], t[f"gold_{c}"], t.requester)])
        t[f"full_{r}_ok"] = full
        t[f"dec_{r}_ok"] = t[f"decision_{r}"] == t.gold_decision
        t[f"amb_{r}"] = t[f"ambiguous_{r}"] == "yes"
    t["disputed"] = (~t.full_a_ok & ~t.full_b_ok) | (t.amb_a & t.amb_b) | (~t.dec_a_ok & ~t.dec_b_ok)
    t.to_csv(OUT / "targeted_audit_items.csv", index=False)

    agr, irr = [], []
    for r in ("a", "b"):
        for typ, g in list(t.groupby("item_type")) + [("all", t)]:
            agr.append({"rater": r, "item_type": typ, "n": len(g), "decision": g[f"dec_{r}_ok"].mean(),
                        "operation_type": g[f"op_{r}_ok"].mean(), "full_operation": g[f"full_{r}_ok"].mean(),
                        "marked_ambiguous": g[f"amb_{r}"].mean()})
        irr.append({"comparison": f"decision rater_{r} vs gold", "kappa": kappa(t[f"decision_{r}"], t.gold_decision)})
        irr.append({"comparison": f"operation rater_{r} vs gold", "kappa": kappa(t[f"operation_{r}"], t.gold_operation)})
    irr.append({"comparison": "decision rater_a vs rater_b", "kappa": kappa(t.decision_a, t.decision_b)})
    irr.append({"comparison": "operation rater_a vs rater_b", "kappa": kappa(t.operation_a, t.operation_b)})
    pd.DataFrame(agr).to_csv(OUT / "targeted_audit_agreement.csv", index=False)
    pd.DataFrame(irr).to_csv(OUT / "targeted_audit_inter_rater.csv", index=False)

    disputed = set(t[t.disputed].instance_id)
    out = pd.read_csv(ROOT / "results" / "governance_analysis" / "instance_level_outcomes.csv",
                      usecols=["system", "pipeline", "instance_id", "unsafe_execution"])
    res = []
    for name, pipe in PIPELINES.items():
        g = out[out.pipeline == pipe]
        for sysname, h in g.groupby("system"):
            keep = h[~h.instance_id.isin(disputed)]
            res.append({"system": sysname, "pipeline": name, "n_all": len(h), "unsafe_all": int(h.unsafe_execution.sum()),
                        "n_without_disputed": len(keep), "unsafe_without_disputed": int(keep.unsafe_execution.sum())})
    res = pd.DataFrame(res)
    res.to_csv(OUT / "residual_failures_without_disputed.csv", index=False)

    summary = {"items": int(len(t)), "targeted": int((t.item_type == "targeted").sum()),
               "disputed_targeted": int((t.disputed & (t.item_type == "targeted")).sum()),
               "disputed_control": int((t.disputed & (t.item_type == "control")).sum()),
               "disputed_behind_dual_failure": int((t.disputed & (t.behind_dual_failure == "True")).sum()),
               "invalid_decision_entries": {r: int((~t[f"decision_{r}"].isin(DECISIONS)).sum()) for r in ("a", "b")}}
    json.dump(summary, open(OUT / "targeted_audit_summary.json", "w"), indent=1)
    print(pd.DataFrame(agr).round(3).to_string(index=False))
    print(pd.DataFrame(irr).round(3).to_string(index=False))
    print(res.groupby("pipeline")[["unsafe_all", "unsafe_without_disputed"]].agg(["min", "max", "sum"]))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
