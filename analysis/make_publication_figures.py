#!/usr/bin/env python3
"""
Publication figure: safety against automation for every system and pipeline.
Input:  results/governance_analysis/metrics_by_stratum.csv (analyze_agent_governance_results.py)
Output: paper/figures/safety_automation_tradeoff.pdf|png
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"


def main():
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    m = pd.read_csv(ROOT / "results" / "governance_analysis" / "metrics_by_stratum.csv")
    a = m[m.stratum == "all"]
    pipes = {"llm_only": ("o", "#d62728", "LLM only"), "validator_gate": ("s", "#1f77b4", "validator gate"),
             "risk_crosscheck": ("^", "#2ca02c", "risk cross-check"),
             "dual_model_agreement|mistral-small3.2:24b": ("D", "#9467bd", "dual-model (Mistral partner)")}
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for p, (mk, c, lab) in pipes.items():
        g = a[a.pipeline == p]
        ax.scatter(g.correct_automation_recall, 100 * g.unsafe_execution_rate, marker=mk, s=14, color=c, alpha=.85,
                   label=lab)
    for b, mk in (("keyword_router", "x"), ("finetuned_encoder", "+")):
        g = a[(a.system == b) & (a.pipeline == "validator_gate")]
        ax.scatter(g.correct_automation_recall, 100 * g.unsafe_execution_rate, marker=mk, s=30, color="k",
                   label=b.replace("_", " ") + " + validator")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 0.5, 1, 5, 10, 50])
    ax.set_yticklabels(["0", "0.5", "1", "5", "10", "50"])
    ax.set_xlabel("correct automation recall")
    ax.set_ylabel("unsafe execution (%)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"safety_automation_tradeoff.{e}", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
