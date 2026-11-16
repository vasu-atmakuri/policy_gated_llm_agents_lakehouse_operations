#!/usr/bin/env python3
"""
Publication figures.
Input:  results/governance_analysis/metrics_by_stratum.csv (analyze_agent_governance_results.py)
Output: paper/figures/safety_automation_tradeoff.pdf|png   safety against automation for every system and pipeline
        paper/figures/unsafe_execution_by_stratum.pdf|png  unsafe execution by system, stratum and pipeline, drawn at
                                                           the printed width of a two-column figure
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


NAMES = {"llama3.2:3b": "Llama 3.2 3B", "qwen3:4b": "Qwen3 4B", "llama3.1:8b": "Llama 3.1 8B", "qwen3:8b": "Qwen3 8B",
         "gemma3:12b": "Gemma 3 12B", "phi4:14b": "Phi-4 14B", "qwen3:14b": "Qwen3 14B", "gpt-oss:20b": "gpt-oss 20B",
         "mistral-small3.2:24b": "Mistral Small 24B", "qwen3:32b": "Qwen3 32B"}
STRATA = [("test_seen_template", "seen"), ("test_unseen_template", "unseen"), ("paraphrased", "paraphrase"),
          ("direct_injection", "direct inj."), ("indirect_injection", "indirect inj."),
          ("false_authority", "false auth."), ("disguise", "disguise"), ("out_of_scope", "out of scope")]


def stratum_heatmap():
    import numpy as np
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "pdf.fonttype": 42})
    m = pd.read_csv(ROOT / "results" / "governance_analysis" / "metrics_by_stratum.csv")
    systems = list(NAMES)
    panels = [("llm_only", "LLM only"), ("validator_gate", "validator gate"), ("risk_crosscheck", "risk cross-check")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), sharey=True)
    for k, (ax, (pipe, title)) in enumerate(zip(axes, panels)):
        g = m[m.pipeline == pipe]
        mat = g.pivot(index="system", columns="stratum", values="unsafe_execution_rate")
        mat = 100 * mat.reindex(index=systems, columns=[s for s, _ in STRATA])
        im = ax.imshow(np.sqrt(mat.values), cmap="Reds", vmin=0, vmax=10, aspect="auto")
        ax.set_xticks(range(len(STRATA)))
        ax.set_xticklabels([lab for _, lab in STRATA], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels([NAMES[s] for s in systems])
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(length=0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    txt = f"{v:.0f}" if pipe == "llm_only" else ("0" if v == 0 else f"{v:.1f}")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=5.8, color="white" if v > 40 else "black")
    fig.tight_layout(w_pad=0.6)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"unsafe_execution_by_stratum.{e}", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
    stratum_heatmap()
