#!/usr/bin/env python3
"""
Additional analyses requested in review of the lakehouse governance paper. Uses only the saved model outputs.

Outputs  results/review_response_analysis/
  paired_attack_outcomes.csv        clean source versus attacked variant, per model x pipeline x attack type
  agreement_definition_ablation.csv dual-model agreement on intent only versus the full normalized operation
  partner_selected_on_dev.csv       partner chosen on the 300 development instances, scored on the evaluation set
  clustered_intervals.csv           unsafe execution with exact counts, Clopper-Pearson upper bounds and
                                    bootstrap intervals clustered by source scenario
  dual_model_latency.csv            per-request latency of one model, two models in sequence and in parallel
  parse_failures.csv                share of agent outputs that could not be parsed
  policy_mutation_detection.csv     validator with one rule removed: unsafe executions it causes on the benchmark
  reasoning_cost.csv                latency and output tokens of Qwen3 with and without reasoning mode
  review_response_summary.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent_evaluation"))
sys.path.insert(0, str(ROOT / "analysis"))

from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.pipelines import (  # noqa: E402
    escalate, llm_decision, normalize_operation, outcome, same_operation, validator_decision,
)
from run_local_model_agent_evaluation import load_instances  # noqa: E402
from analyze_agent_governance_results import BASELINES, stratum  # noqa: E402

OUTPUTS = ROOT / "results" / "agent_outputs"
OUT = ROOT / "results" / "review_response_analysis"
RNG = np.random.default_rng(20261116)
N_BOOT = 2000
ATTACKS = ["direct_injection", "indirect_injection", "false_authority", "disguise"]
PARTNERS = ["mistral-small3.2:24b", "gemma3:12b"]


def load():
    parsed, latency = defaultdict(dict), defaultdict(dict)
    for p in sorted(OUTPUTS.glob("*.jsonl")):
        for line in p.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["call"] != "agent":
                continue
            parsed[r["model"]][r["instance_id"]] = r.get("parsed")
            if r.get("latency_s") is not None:
                latency[r["model"]][r["instance_id"]] = r["latency_s"]
    return parsed, latency


def cp_upper(k, n):
    return 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))


def cp_lower(k, n):
    return 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))


def cluster_boot(values, clusters):
    df = pd.DataFrame({"v": values, "c": clusters}).groupby("c").v.agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    idx = RNG.integers(0, len(s), size=(N_BOOT, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def decide(pipeline, op, parsed, inst, env, partner_op=None, intent_only=False):
    if pipeline == "llm_only":
        return llm_decision(parsed)
    v = validator_decision(op, inst["context"], env)
    if pipeline == "validator_gate":
        return v
    if intent_only:
        agree = op is not None and partner_op is not None and op["intent"] == partner_op["intent"]
    else:
        agree = same_operation(op, partner_op)
    return v if agree else escalate(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = build_environment()
    rows = load_instances()
    manifest = json.load(open(ROOT / "data" / "evaluation_manifest.json"))
    eval_ids = manifest["evaluation_instance_ids"] + [i for i, r in rows.items() if r.get("split") == "paraphrased"]
    dev_ids = manifest["threshold_tuning_instance_ids"]
    eval_set = set(eval_ids)
    parsed, latency = load()
    llms = [m for m in parsed if m not in BASELINES]
    ops = {m: {i: normalize_operation(parsed[m].get(i), rows[i]["context"]["requester"])
               for i in set(eval_ids) | set(dev_ids) if i in parsed[m]} for m in parsed}
    summary = {}

    # 1. paired clean versus attacked outcomes
    pairs = [(i, rows[i]["source_instance_id"]) for i in eval_ids
             if rows[i]["attack_type"] in ATTACKS and rows[i].get("source_instance_id") in eval_set]
    recs = []
    for m in llms:
        for pipe in ["llm_only", "validator_gate"]:
            for atk, sub in [("all_attacks", pairs)] + [(a, [p for p in pairs if rows[p[0]]["attack_type"] == a]) for a in ATTACKS]:
                clean_u, att_u = [], []
                for a, s in sub:
                    if a not in parsed[m] or s not in parsed[m]:
                        continue
                    for iid, store in ((s, clean_u), (a, att_u)):
                        d = decide(pipe, ops[m][iid], parsed[m][iid], rows[iid], env)
                        store.append(outcome(rows[iid], d, ops[m][iid])["unsafe_execution"])
                cu, au = np.array(clean_u), np.array(att_u)
                recs.append({"system": m, "pipeline": pipe, "attack_type": atk, "n_pairs": len(cu),
                             "same_gold_decision": int(sum(rows[a]["gold_decision"] == rows[s]["gold_decision"] for a, s in sub)),
                             "unsafe_clean": cu.mean(), "unsafe_attacked": au.mean(),
                             "safe_to_unsafe": int(np.sum(~cu & au)), "unsafe_to_safe": int(np.sum(cu & ~au)),
                             "attack_success_rate": np.sum(~cu & au) / max(1, np.sum(~cu))})
    paired = pd.DataFrame(recs)
    paired.to_csv(OUT / "paired_attack_outcomes.csv", index=False)
    summary["paired_pairs"] = len(pairs)

    # 2. agreement definition ablation and 3. partner chosen on development instances
    def dual(m, b, ids, intent_only=False):
        u, auto = [], []
        for iid in ids:
            if iid not in ops[m] or iid not in ops[b]:
                continue
            d = decide("dual", ops[m][iid], parsed[m][iid], rows[iid], env, ops[b][iid], intent_only)
            o = outcome(rows[iid], d, ops[m][iid])
            u.append(o["unsafe_execution"])
            auto.append(d == "auto_execute")
        return np.array(u), np.array(auto)

    abl = []
    for m in llms:
        for b in PARTNERS:
            if b == m:
                continue
            for kind, io in (("full_operation", False), ("intent_only", True)):
                u, a = dual(m, b, eval_ids, io)
                abl.append({"system": m, "partner": b, "agreement": kind, "n": len(u), "unsafe_count": int(u.sum()),
                            "unsafe_rate": u.mean(), "automation_rate": a.mean()})
    pd.DataFrame(abl).to_csv(OUT / "agreement_definition_ablation.csv", index=False)

    sel = []
    for m in llms:
        cands = []
        for b in llms:
            if b == m:
                continue
            u, a = dual(m, b, dev_ids)
            cands.append((u.mean(), -a.mean(), b))
        du, na, best = sorted(cands)[0]
        u, a = dual(m, best, eval_ids)
        sel.append({"system": m, "partner_selected_on_dev": best, "dev_unsafe": du, "dev_automation": -na,
                    "eval_n": len(u), "eval_unsafe_count": int(u.sum()), "eval_unsafe_rate": u.mean(),
                    "eval_unsafe_cp_upper": cp_upper(int(u.sum()), len(u)), "eval_automation": a.mean()})
    pd.DataFrame(sel).to_csv(OUT / "partner_selected_on_dev.csv", index=False)

    # 4. exact counts and clustered intervals for the main pipelines
    src = {i: rows[i].get("source_instance_id") or i for i in eval_ids}
    ci = []
    for m in parsed:
        pipes = ["validator_gate"] if m in BASELINES else ["llm_only", "validator_gate", "dual|mistral-small3.2:24b"]
        for pipe in pipes:
            if pipe.startswith("dual") and m == "mistral-small3.2:24b":
                continue
            u, clus = [], []
            for iid in eval_ids:
                if iid not in parsed[m]:
                    continue
                if pipe.startswith("dual"):
                    if iid not in ops["mistral-small3.2:24b"]:
                        continue
                    d = decide("dual", ops[m][iid], parsed[m][iid], rows[iid], env, ops["mistral-small3.2:24b"][iid])
                else:
                    d = decide(pipe, ops[m][iid], parsed[m][iid], rows[iid], env)
                u.append(outcome(rows[iid], d, ops[m][iid])["unsafe_execution"])
                clus.append(src[iid])
            u = np.array(u)
            k, n = int(u.sum()), len(u)
            lo, hi = cluster_boot(u.astype(float), clus)
            ci.append({"system": m, "pipeline": pipe, "n": n, "unsafe_count": k, "unsafe_rate": k / n,
                       "cp_lower": cp_lower(k, n), "cp_upper": cp_upper(k, n), "cluster_boot_lo": lo, "cluster_boot_hi": hi,
                       "n_clusters": len(set(clus))})
    pd.DataFrame(ci).to_csv(OUT / "clustered_intervals.csv", index=False)

    # 5. latency of dual-model agreement with the Mistral partner
    lat = []
    part = "mistral-small3.2:24b"
    for m in llms:
        ids = [i for i in eval_ids if i in latency[m]]
        one = np.array([latency[m][i] for i in ids])
        both = [i for i in ids if i in latency[part]]
        seq = np.array([latency[m][i] + latency[part][i] for i in both])
        par = np.array([max(latency[m][i], latency[part][i]) for i in both])
        lat.append({"system": m, "median_single_s": np.median(one), "median_dual_sequential_s": np.median(seq),
                    "median_dual_parallel_s": np.median(par), "p95_single_s": np.quantile(one, .95),
                    "p95_dual_parallel_s": np.quantile(par, .95)})
    pd.DataFrame(lat).to_csv(OUT / "dual_model_latency.csv", index=False)

    # 6. parse failures on the evaluation set
    pf = [{"system": m, "n": sum(i in parsed[m] for i in eval_ids),
           "parse_failures": sum(parsed[m][i] is None for i in eval_ids if i in parsed[m]),
           "out_of_scope_or_unknown_intent": sum(ops[m][i] is not None and ops[m][i]["intent"] == "out_of_scope"
                                                 and rows[i]["operation"]["intent"] != "out_of_scope"
                                                 for i in eval_ids if i in ops[m])} for m in llms]
    pf = pd.DataFrame(pf)
    pf["parse_failure_rate"] = pf.parse_failures / pf.n
    pf.to_csv(OUT / "parse_failures.csv", index=False)
    # 7. policy mutation: remove one rule from the validator, keep gold labels from the full policy
    import lakehouse_governance_benchmark.policy as policy
    orig = policy._result
    mut = []
    part = "mistral-small3.2:24b"
    oracle = {i: (normalize_operation(rows[i]["operation"], rows[i]["context"]["requester"])
                  if rows[i]["operation"]["intent"] != "out_of_scope" else rows[i]["operation"]) for i in eval_ids}
    for rule in sorted(policy.RULES):
        policy._result = lambda fired, r=rule: orig({k: v for k, v in fired.items() if k != r})
        k_oracle = k_val = k_dual = 0
        for iid in eval_ids:
            inst = rows[iid]
            k_oracle += outcome(inst, validator_decision(oracle[iid], inst["context"], env), oracle[iid])["unsafe_execution"]
            if iid in ops[part] and iid in ops["gemma3:12b"]:
                v = validator_decision(ops[part][iid], inst["context"], env)
                k_val += outcome(inst, v, ops[part][iid])["unsafe_execution"]
                d = v if same_operation(ops[part][iid], ops["gemma3:12b"][iid]) else escalate(v)
                k_dual += outcome(inst, d, ops[part][iid])["unsafe_execution"]
        mut.append({"removed_rule": rule, "description": policy.RULES[rule], "unsafe_with_gold_operations": k_oracle,
                    "unsafe_validator_mistral": k_val, "unsafe_dual_mistral_gemma": k_dual})
    policy._result = orig
    mut = pd.DataFrame(mut)
    mut.to_csv(OUT / "policy_mutation_detection.csv", index=False)
    summary["mutations_detected_with_gold_operations"] = int((mut.unsafe_with_gold_operations > 0).sum())

    # 8. cost of reasoning mode on the 800-instance ablation subset
    abl_dir = ROOT / "results" / "agent_ablation_outputs"
    sub = set(json.load(open(abl_dir / "subset_instance_ids.json")))
    rc = []
    for size in ["4b", "8b", "14b", "32b"]:
        recs = [json.loads(x) for x in open(abl_dir / f"qwen3_{size}__reasoning.jsonl")]
        recs = [r for r in recs if r["call"] == "agent" and r.get("latency_s") is not None]
        base = [json.loads(x) for x in open(OUTPUTS / f"qwen3_{size}.jsonl")]
        base = [r for r in base if r["call"] == "agent" and r["instance_id"] in sub and r.get("latency_s") is not None]
        rc.append({"system": f"qwen3:{size}", "median_latency_s": np.median([r["latency_s"] for r in base]),
                   "median_latency_reasoning_s": np.median([r["latency_s"] for r in recs]),
                   "median_output_tokens": np.median([r.get("output_tokens") or 0 for r in base]),
                   "median_output_tokens_reasoning": np.median([r.get("output_tokens") or 0 for r in recs])})
    pd.DataFrame(rc).to_csv(OUT / "reasoning_cost.csv", index=False)

    json.dump(summary, open(OUT / "review_response_summary.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
