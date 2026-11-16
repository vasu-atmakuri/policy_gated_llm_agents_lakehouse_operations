#!/usr/bin/env python3
"""
Scores the returned human validation sheets against the benchmark's gold decisions and gold operations.

Input: a folder holding the four returned sheets, named as they were sent out:
  policy_label_sheet_rater_a.xlsx, policy_label_sheet_rater_b.xlsx,
  operation_label_sheet_rater_a.xlsx, operation_label_sheet_rater_b.xlsx
The raw files are not copied into the repository. Only the answer columns are kept, without file metadata,
in data/human_policy_validation/returned_responses/.

Outputs (results/human_validation_analysis/):
  decision_agreement.csv        per rater and stratum: agreement with the gold decision
  operation_agreement.csv       per rater and stratum: agreement on operation type, each argument and the full operation
  inter_rater_agreement.csv     Cohen's kappa between raters and between each rater and gold
  items_for_adjudication.csv    items where at least one rater differs from gold, with both answers
  human_validation_summary.json
Usage: python analysis/score_human_validation_sheets.py /path/to/returned_sheets
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "human_policy_validation"
OUT = ROOT / "results" / "human_validation_analysis"
DECISIONS = ["auto_execute", "human_review", "reject"]
ARGS = ["resource", "principal", "number", "option"]


def kappa(a, b):
    a, b = pd.Series(a).astype(str), pd.Series(b).astype(str)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a.values == b.values))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def norm_text(s):
    return " ".join(clean(s).lower().replace('"', "'").strip("`'").split())


def norm_predicate(s):
    return re.sub(r"[\s'\"`;]", "", clean(s).lower())


def arg_match(col, rater, gold, requester):
    r, g = clean(rater), clean(gold)
    if col == "principal" and norm_text(r) in ("requester", "me", "myself", "self"):
        r = requester
    if col == "number":
        try:
            return int(float(r)) == int(float(g)) if g else r == ""
        except ValueError:
            return False
    if col == "option" and ("=" in g or "=" in r):
        return norm_predicate(r) == norm_predicate(g)
    return norm_text(r) == norm_text(g)


def read_sheet(folder, name, cols):
    df = pd.read_excel(folder / name, sheet_name="Items", dtype=str)
    missing = [c for c in ["item_id"] + cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name}: missing columns {missing}")
    df = df[["item_id"] + cols].copy()
    df["item_id"] = df.item_id.astype(float).astype(int)
    for c in cols:
        df[c] = df[c].map(clean)
    return df


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "returned_responses").mkdir(parents=True, exist_ok=True)
    pkey = pd.read_csv(DATA / "policy_label_answer_key.csv")
    okey = pd.read_csv(DATA / "operation_label_answer_key.csv", dtype=str, keep_default_na=False)
    okey["item_id"] = okey.item_id.astype(int)
    summary, dec_rows, op_rows, irr, adj = {}, [], [], [], {}

    pol, opr = {}, {}
    for r in ("a", "b"):
        pol[r] = read_sheet(folder, f"policy_label_sheet_rater_{r}.xlsx", ["decision", "notes"])
        pol[r]["decision"] = pol[r].decision.str.strip().str.lower()
        opr[r] = read_sheet(folder, f"operation_label_sheet_rater_{r}.xlsx",
                            ["operation", "resource", "principal", "number", "option", "ambiguous", "notes"])
        opr[r]["operation"] = opr[r].operation.str.strip().str.lower()
        pol[r].to_csv(DATA / "returned_responses" / f"policy_label_responses_rater_{r}.csv", index=False)
        opr[r].to_csv(DATA / "returned_responses" / f"operation_label_responses_rater_{r}.csv", index=False)

    # decisions
    d = pkey.copy()
    for r in ("a", "b"):
        d = d.merge(pol[r][["item_id", "decision"]].rename(columns={"decision": f"decision_{r}"}), on="item_id")
        d[f"decision_{r}_ok"] = d[f"decision_{r}"] == d.gold_decision
        summary[f"decision_invalid_entries_{r}"] = int((~d[f"decision_{r}"].isin(DECISIONS)).sum())
    for r in ("a", "b"):
        for st, g in list(d.groupby("stratum")) + [("all", d)]:
            dec_rows.append({"rater": r, "stratum": st, "n": len(g), "agree_with_gold": g[f"decision_{r}_ok"].mean()})
        irr.append({"comparison": f"decision rater_{r} vs gold", "kappa": kappa(d[f"decision_{r}"], d.gold_decision),
                    "raw_agreement": d[f"decision_{r}_ok"].mean()})
    irr.append({"comparison": "decision rater_a vs rater_b", "kappa": kappa(d.decision_a, d.decision_b),
                "raw_agreement": float(np.mean(d.decision_a == d.decision_b))})
    summary["decision_both_raters_agree_with_each_other_not_gold"] = int(
        ((d.decision_a == d.decision_b) & ~d.decision_a_ok).sum())

    # operations
    o = okey.copy()
    for r in ("a", "b"):
        x = opr[r].rename(columns={c: f"{c}_{r}" for c in ["operation"] + ARGS + ["ambiguous", "notes"]})
        o = o.merge(x, on="item_id")
        o[f"op_{r}_ok"] = o[f"operation_{r}"] == o.gold_operation
        full = o[f"op_{r}_ok"].copy()
        for c in ARGS:
            o[f"{c}_{r}_ok"] = [arg_match(c, rv, gv, req) for rv, gv, req in zip(o[f"{c}_{r}"], o[f"gold_{c}"], o.requester)]
            full &= o[f"{c}_{r}_ok"]
        o[f"full_{r}_ok"] = full
    for r in ("a", "b"):
        for st, g in list(o.groupby("stratum")) + [("all", o)]:
            row = {"rater": r, "stratum": st, "n": len(g), "operation_type": g[f"op_{r}_ok"].mean(),
                   "full_operation": g[f"full_{r}_ok"].mean(),
                   "marked_ambiguous": float(np.mean(g[f"ambiguous_{r}"].str.lower() == "yes"))}
            same = g[g[f"op_{r}_ok"]]
            for c in ARGS:
                applies = same[same[f"gold_{c}"] != ""]
                row[f"{c}_given_type"] = applies[f"{c}_{r}_ok"].mean() if len(applies) else np.nan
            op_rows.append(row)
        irr.append({"comparison": f"operation rater_{r} vs gold", "kappa": kappa(o[f"operation_{r}"], o.gold_operation),
                    "raw_agreement": o[f"op_{r}_ok"].mean()})
    irr.append({"comparison": "operation rater_a vs rater_b", "kappa": kappa(o.operation_a, o.operation_b),
                "raw_agreement": float(np.mean(o.operation_a == o.operation_b))})
    summary["operation_both_raters_agree_with_each_other_not_gold"] = int(
        ((o.operation_a == o.operation_b) & ~o.op_a_ok).sum())
    summary["full_operation_both_raters_differ_from_gold"] = int((~o.full_a_ok & ~o.full_b_ok).sum())

    # items for adjudication
    flag = d[~d.decision_a_ok | ~d.decision_b_ok][["item_id"]].assign(decision_issue=True)
    flag = flag.merge(o[~o.full_a_ok | ~o.full_b_ok][["item_id"]].assign(operation_issue=True), on="item_id", how="outer")
    cols = ["item_id", "instance_id", "stratum", "gold_decision", "decision_a", "decision_b"]
    ocols = ["item_id", "gold_operation", "gold_resource", "gold_principal", "gold_number", "gold_option"] + [
        f"{c}_{r}" for r in ("a", "b") for c in ["operation"] + ARGS + ["ambiguous", "notes"]]
    adj = flag.merge(d[cols], on="item_id").merge(o[ocols], on="item_id").sort_values("item_id")
    adj.to_csv(OUT / "items_for_adjudication.csv", index=False)

    pd.DataFrame(dec_rows).to_csv(OUT / "decision_agreement.csv", index=False)
    pd.DataFrame(op_rows).to_csv(OUT / "operation_agreement.csv", index=False)
    pd.DataFrame(irr).to_csv(OUT / "inter_rater_agreement.csv", index=False)
    summary["items"] = int(len(d))
    summary["items_for_adjudication"] = int(len(adj))
    json.dump(summary, open(OUT / "human_validation_summary.json", "w"), indent=1)
    print(pd.DataFrame(irr).round(3).to_string(index=False))
    print(pd.DataFrame(op_rows).query("stratum == 'all'").round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
