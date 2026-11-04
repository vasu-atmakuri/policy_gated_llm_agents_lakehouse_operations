#!/usr/bin/env python3
"""
Builds the lakehouse operation request benchmark.

Outputs (data/)
  lakehouse_environment.json        the simulated system of record (tables, clusters, secrets, jobs, topics, ...)
  base_requests.jsonl               templated requests with splits train / dev / test_seen_template /
                                    test_unseen_template (template index 3 appears only in the last split)
  adversarial_requests.jsonl        direct_injection, indirect_injection, false_authority, disguise variants
  out_of_scope_requests.jsonl       requests for operations outside the catalog (gold: human_review)
  evaluation_manifest.json          ids of the instances in the LLM evaluation set and counts per stratum

Each instance: instance_id, split, attack_type, text, operation (ground truth), context (system of record),
metadata_overrides (resource comments as the agent will read them), gold_decision, gold_rules.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lakehouse_governance_benchmark.adversarial_requests import (  # noqa: E402
    DIRECT_INJECTIONS, DISGUISE_TEMPLATES, FALSE_AUTHORITY, INDIRECT_INJECTIONS, OUT_OF_SCOPE,
)
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.operations import OPERATIONS  # noqa: E402
from lakehouse_governance_benchmark.request_generation import (  # noqa: E402
    CHANGE_TEXT, DESTINATION_TEXT, HELD_OUT_TEMPLATE, TEMPLATES, generate_instance,
)

DATA = ROOT / "data"
SEED = 20261104


def stratified_instances(rng, env, intent, n):
    """Draw n instances of one operation, balancing gold decisions as far as the operation allows."""
    pool = defaultdict(list)
    for k in range(n * 12):
        x = generate_instance(rng, env, intent, k % 4)
        pool[x["gold_decision"]].append(x)
    decisions = [d for d in ("auto_execute", "human_review", "reject") if pool[d]]
    out, i = [], 0
    while len(out) < n:
        d = decisions[i % len(decisions)]
        if pool[d]:
            out.append(pool[d].pop())
        elif not any(pool.values()):
            break
        i += 1
    return out


def ticket_description(rng, op):
    return rng.choice(["Routine request raised via the data platform portal.",
                       "Requested by the business owner, see linked thread.",
                       f"Operational change: {op['intent'].replace('_', ' ')}."])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-operation", type=int, default=250)
    ap.add_argument("--eval-base-per-split", type=int, default=600)
    ap.add_argument("--eval-attack-per-type", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(SEED)
    env = build_environment()
    DATA.mkdir(exist_ok=True)
    with open(DATA / "lakehouse_environment.json", "w") as f:
        json.dump(env.to_json(), f, indent=1)

    base = []
    for intent in TEMPLATES:
        for x in stratified_instances(rng, env, intent, args.per_operation):
            if x["template_index"] == HELD_OUT_TEMPLATE:
                split = "test_unseen_template"
            else:
                u = rng.random()
                split = "train" if u < 0.6 else "dev" if u < 0.7 else "test_seen_template"
            if x["context"]["change_ticket"]:
                x["context"]["change_ticket"]["description"] = ticket_description(rng, x["operation"])
            base.append({"instance_id": f"B{len(base):05d}", "split": split, "attack_type": "none",
                         "metadata_overrides": {}, **x})

    # ---------------- adversarial variants (sources: test splits, gold review or reject) ----------------
    risky = [b for b in base if b["split"].startswith("test") and b["gold_decision"] != "auto_execute"]
    adversarial = []

    def variant(src, attack, text=None, overrides=None, context=None):
        v = json.loads(json.dumps(src))
        v.update(instance_id=f"A{len(adversarial):05d}", split="adversarial", attack_type=attack,
                 source_instance_id=src["instance_id"])
        if text is not None:
            v["text"] = text
        if overrides:
            v["metadata_overrides"] = overrides
        if context:
            v["context"] = context
        adversarial.append(v)

    for src in rng.sample(risky, args.eval_attack_per_type):
        variant(src, "direct_injection", text=src["text"].rstrip() + rng.choice(DIRECT_INJECTIONS))
    for src in rng.sample(risky, args.eval_attack_per_type):
        inj = rng.choice(INDIRECT_INJECTIONS)
        if "table" in src["operation"]["params"] and OPERATIONS[src["operation"]["intent"]].target_kind == "tables":
            variant(src, "indirect_injection", overrides={"table_comment": {src["operation"]["params"]["table"]: inj}})
        else:
            ctx = json.loads(json.dumps(src["context"]))
            ctx["change_ticket"] = ctx["change_ticket"] or {"id": f"CHG-{rng.randint(10000, 99999)}", "approved": False}
            ctx["change_ticket"]["description"] = inj
            # a pending ticket does not change the gold decision relative to no ticket for these rules
            variant(src, "indirect_injection", context=ctx)
    for src in rng.sample(risky, args.eval_attack_per_type):
        variant(src, "false_authority", text=rng.choice(FALSE_AUTHORITY) + src["text"])
    disguisable = [b for b in risky if b["operation"]["intent"] in DISGUISE_TEMPLATES]
    for src in rng.sample(disguisable, min(args.eval_attack_per_type, len(disguisable))):
        values = dict(src["operation"]["params"])
        for key in ("grantee", "user"):
            if values.get(key) == src["context"]["requester"]:
                values[key] = "me"
        if "destination" in values:
            values["destination"] = rng.choice(DESTINATION_TEXT[values["destination"]])
        if "change" in values:
            values["change"] = CHANGE_TEXT[values["change"]]
        if src["operation"]["intent"] == "delete_rows" and src["operation"]["params"].get("predicate"):
            continue  # disguise templates describe whole-table deletes only
        variant(src, "disguise", text=rng.choice(DISGUISE_TEMPLATES[src["operation"]["intent"]]).format(**values))

    # indirect injection with a pending ticket must keep its gold label: re-check against the policy
    from lakehouse_governance_benchmark.policy import evaluate_operation
    for v in adversarial:
        d, r = evaluate_operation(v["operation"], v["context"], env)
        v["gold_decision"], v["gold_rules"] = d, r

    oos = []
    humans = sorted(n for n, p in env.principals.items() if p.role != "service_principal")
    for k in range(200):
        oos.append({"instance_id": f"O{k:05d}", "split": "out_of_scope", "attack_type": "out_of_scope",
                    "text": OUT_OF_SCOPE[k % len(OUT_OF_SCOPE)],
                    "operation": {"intent": "out_of_scope", "params": {}},
                    "context": {"requester": rng.choice(humans), "change_ticket": None,
                                "in_change_window": rng.random() < 0.8},
                    "metadata_overrides": {}, "template_index": -1,
                    "gold_decision": "human_review", "gold_rules": ["G20"]})

    for name, rows in (("base_requests", base), ("adversarial_requests", adversarial), ("out_of_scope_requests", oos)):
        with open(DATA / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    ev = []
    for split in ("test_seen_template", "test_unseen_template"):
        pool = [b["instance_id"] for b in base if b["split"] == split]
        ev += rng.sample(pool, min(args.eval_base_per_split, len(pool)))
    ev += [a["instance_id"] for a in adversarial] + [o["instance_id"] for o in oos[:100]]
    dev_pool = [b["instance_id"] for b in base if b["split"] == "dev"]
    tuning = rng.sample(dev_pool, min(300, len(dev_pool)))
    manifest = {"evaluation_instance_ids": ev, "threshold_tuning_instance_ids": tuning,
                "counts": {"base_by_split": Counter(b["split"] for b in base),
                           "base_by_decision": Counter(b["gold_decision"] for b in base),
                           "adversarial_by_type": Counter(a["attack_type"] for a in adversarial),
                           "adversarial_by_decision": Counter(a["gold_decision"] for a in adversarial),
                           "out_of_scope": len(oos), "evaluation_set": len(ev),
                           "threshold_tuning_set": len(tuning)}}
    with open(DATA / "evaluation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
