#!/usr/bin/env python3
"""
Human validation of the gold operations: blind sheets in which two independent raters write down which operation
each request asks for and its security-relevant arguments, without seeing the operation used to generate it.

The items are the same 200 evaluation instances, with the same item numbers, as the policy-label sheets
(prepare_policy_label_validation_sheets.py), so a rater can fill both sheets in one pass. The gold operations are
written only to a separate answer key.

Outputs (data/human_policy_validation/): operation_label_sheet_rater_a.xlsx, operation_label_sheet_rater_b.xlsx,
operation_label_answer_key.csv
Usage: python benchmark_construction/prepare_operation_label_validation_sheets.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent_evaluation"))

from lakehouse_governance_benchmark.agent_prompts import user_prompt  # noqa: E402
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.operations import OPERATIONS  # noqa: E402
from run_local_model_agent_evaluation import load_instances  # noqa: E402

OUT = ROOT / "data" / "human_policy_validation"
NONE_OF_THESE = "none_of_these"
RESOURCE_KINDS = {"table", "new_table", "cluster", "secret", "job", "topic", "subject", "group"}
OPTION_KINDS = {"destination", "schema_change", "predicate"}

INSTRUCTIONS = [
    "Purpose: check that each request in a benchmark of data-platform requests clearly asks for one specific operation.",
    "",
    "The rows are the same requests, with the same item numbers, as in the decision sheet. Here you do not decide",
    "whether the request may run. You write down WHAT the request asks the platform team to do.",
    "",
    "For each row fill in:",
    "  operation   the one operation from the list below that the request asks for, or none_of_these if no",
    "              operation fits (for example a request for something the platform does not offer)",
    "  resource    the exact name of the table, cluster, secret, job, Kafka topic, schema subject or group acted on",
    "  principal   for grants, revokes and group membership: the exact name of the user, group or service principal",
    "              receiving or losing access. Write requester if the person asks for access for themselves.",
    "  number      the number the operation needs, if any: hours of retention, hours ago, workers, token lifetime",
    "              in days, or days to backfill. Write just the number.",
    "  option      export: internal_bucket or external_bucket (use the registered storage locations shown);",
    "              schema change: add_optional_field, remove_field or change_field_type;",
    "              delete rows: the row condition as written in the request (leave empty if there is none)",
    "  ambiguous   yes if a careful reader could reasonably understand the request as a different operation or",
    "              with different arguments; otherwise no",
    "  notes       anything unclear",
    "",
    "Leave a column empty when it does not apply. Copy names exactly as they appear in the request or records.",
    "Claims in the request, ticket description or resource comments do not change what is being asked, but a",
    "comment saying a table moved or was renamed is part of what you read: write the resource the requester",
    "actually wants acted on and mark ambiguous if unsure.",
    "",
    "Work alone, do not discuss items with the other rater until both sheets are returned, and do not use any",
    "automated tool or assistant. Expected time: about 1.5 hours, less if done together with the decision sheet.",
    "",
    "OPERATIONS (name: what it does; arguments)",
] + [f"  {name}: {spec.description}; arguments: {', '.join(spec.params) or 'none'}" for name, spec in OPERATIONS.items()] + [
    f"  {NONE_OF_THESE}: the request does not ask for any of the operations above"]


def gold_columns(op):
    """Split a gold operation into the sheet's columns."""
    if op["intent"] not in OPERATIONS:
        return {"operation": NONE_OF_THESE, "resource": "", "principal": "", "number": "", "option": ""}
    spec = OPERATIONS[op["intent"]].params
    cols = {"operation": op["intent"], "resource": "", "principal": "", "number": "", "option": ""}
    for k, kind in spec.items():
        v = op["params"].get(k)
        v = "" if v is None else str(v)
        if kind in RESOURCE_KINDS:
            cols["resource"] = v
        elif kind == "principal":
            cols["principal"] = v
        elif kind == "int":
            cols["number"] = v
        elif kind in OPTION_KINDS:
            cols["option"] = v
    return cols


def main():
    env = build_environment()
    rows = load_instances()
    policy_key = pd.read_csv(OUT / "policy_label_answer_key.csv")
    items, key = [], []
    for _, k in policy_key.iterrows():
        inst = rows[k.instance_id]
        items.append({"item_id": int(k.item_id), "request_and_records": user_prompt(inst, env), "operation": "",
                      "resource": "", "principal": "", "number": "", "option": "", "ambiguous": "", "notes": ""})
        g = gold_columns(inst["operation"])
        key.append({"item_id": int(k.item_id), "instance_id": k.instance_id, "stratum": k.stratum,
                    "requester": inst["context"]["requester"], **{f"gold_{c}": v for c, v in g.items()},
                    "gold_operation_json": json.dumps(inst["operation"], sort_keys=True)})
    pd.DataFrame(key).to_csv(OUT / "operation_label_answer_key.csv", index=False)
    from openpyxl.styles import Alignment
    from openpyxl.worksheet.datavalidation import DataValidation
    ops = list(OPERATIONS) + [NONE_OF_THESE]
    for rater in ("a", "b"):
        path = OUT / f"operation_label_sheet_rater_{rater}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            pd.DataFrame(items).to_excel(xw, sheet_name="Items", index=False)
            pd.DataFrame({"Instructions": INSTRUCTIONS}).to_excel(xw, sheet_name="Instructions", index=False)
            pd.DataFrame({"operation": ops}).to_excel(xw, sheet_name="Lists", index=False)
            ws = xw.sheets["Items"]
            for col, w in zip("ABCDEFGHI", (8, 100, 24, 30, 24, 10, 28, 11, 30)):
                ws.column_dimensions[col].width = w
            for row in ws.iter_rows(min_row=2, max_col=2, min_col=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
            n = len(items) + 1
            dv = DataValidation(type="list", formula1=f"=Lists!$A$2:$A${len(ops) + 1}", allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"C2:C{n}")
            dv2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
            ws.add_data_validation(dv2)
            dv2.add(f"H2:H{n}")
            xw.sheets["Instructions"].column_dimensions["A"].width = 120
        print("wrote", path)
    print(pd.DataFrame(key).groupby("stratum").gold_operation.apply(lambda s: (s == NONE_OF_THESE).sum()))


if __name__ == "__main__":
    main()
