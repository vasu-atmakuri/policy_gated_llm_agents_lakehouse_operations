#!/usr/bin/env python3
"""
Supervised baseline: a fine-tuned transformer encoder (default distilroberta-base) classifies the operation type
(24 operations plus out_of_scope); parameters come from the deterministic parser; every proposed operation goes
through the codified validator. Trained on the train split (templates 0 to 2 only), early stopping on dev.

Writes results/agent_outputs/finetuned_encoder.jsonl in the same format as the local-model runs.
Usage: python agent_evaluation/run_finetuned_encoder_baseline.py --epochs 4
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent_evaluation"))

from lakehouse_governance_benchmark.deterministic_parser import extract_parameters  # noqa: E402
from lakehouse_governance_benchmark.environment import build_environment  # noqa: E402
from lakehouse_governance_benchmark.operations import OPERATIONS  # noqa: E402
from lakehouse_governance_benchmark.pipelines import normalize_operation, validator_decision  # noqa: E402
from run_local_model_agent_evaluation import OUT, evaluation_ids, load_instances  # noqa: E402

LABELS = list(OPERATIONS) + ["out_of_scope"]


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = build_environment()
    rows = load_instances()
    train = [r for r in rows.values() if r["split"] == "train"]
    # out-of-scope examples for training are written separately from the evaluation phrasing
    oos_train = ["please change the company's cloud billing plan", "can you make me an owner of the whole workspace",
                 "turn off logging for our cluster permanently", "set up a VPN for our partner",
                 "create a new metastore in another region", "remove all users from the workspace"]
    texts = [r["text"] for r in train] + oos_train * 10
    labels = [LABELS.index(r["operation"]["intent"]) for r in train] + [LABELS.index("out_of_scope")] * (10 * len(oos_train))
    dev = [r for r in rows.values() if r["split"] == "dev"]

    dev_ = device()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS)).to(dev_)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(texts) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    def predict(batch_texts):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(batch_texts), 64):
                enc = tok(batch_texts[i:i + 64], truncation=True, max_length=96, padding=True, return_tensors="pt").to(dev_)
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    best_acc, best_state = -1, None
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(texts))
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=96, padding=True, return_tensors="pt").to(dev_)
            loss = model(**enc, labels=torch.tensor([labels[j] for j in idx]).to(dev_)).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
        probs = predict([r["text"] for r in dev])
        acc = np.mean([LABELS[k] == r["operation"]["intent"] for k, r in zip(probs.argmax(1), dev)])
        print(f"epoch {epoch + 1}: dev intent accuracy {acc:.3f}", flush=True)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    ids = evaluation_ids(rows)
    probs = predict([rows[i]["text"] for i in ids])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "finetuned_encoder.jsonl", "w") as f:
        for iid, pr in zip(ids, probs):
            inst = rows[iid]
            intent = LABELS[int(pr.argmax())]
            params = extract_parameters(inst["text"], intent, inst["context"]["requester"], env)
            op = normalize_operation({"intent": intent, "params": params}, inst["context"]["requester"])
            parsed = {"intent": intent, "params": params, "decision": validator_decision(op, inst["context"], env),
                      "confidence": float(pr.max()), "rationale": "fine-tuned encoder"}
            f.write(json.dumps({"model": "finetuned_encoder", "instance_id": iid, "call": "agent", "parsed": parsed}) + "\n")
    print(f"finetuned encoder outputs written (best dev intent accuracy {best_acc:.3f})")


if __name__ == "__main__":
    main()
