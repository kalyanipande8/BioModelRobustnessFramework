"""
run_all.py

Orchestrates the full pipeline:
  1. generate_eval_conditions.py  (base dataset + 89 conditions)
  2. train_model_variants.py      (18 model variants)
  3. run_evaluation_suite.py      (18 x 89 = 1602 real eval runs + failure analysis)
  4. Builds figures (heatmap, robustness bar chart, failure rate by category)
  5. Writes output/report.md with real numbers/findings from the actual run.
"""
import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def run_step(script):
    print(f"\n{'='*70}\nRunning {script}\n{'='*70}")
    result = subprocess.run([sys.executable, script], cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def make_figures(matrix, manifest, analysis, summary):
    os.makedirs(FIG_DIR, exist_ok=True)
    manifest_by_id = {m["id"]: m for m in manifest}

    # Figure 1: variant x condition heatmap
    fig, ax = plt.subplots(figsize=(20, 7))
    im = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{vid} ({manifest_by_id[vid]['name']})" for vid in matrix.index],
                        fontsize=6)
    ax.set_xticks([])
    ax.set_xlabel("89 evaluation conditions (grouped: noise | ood_shift | "
                   "class_imbalance | adversarial | missing_feature)")
    ax.set_title("Variant x Condition Accuracy Matrix (18 variants x 89 conditions)")
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "variant_condition_heatmap.png"), dpi=130)
    plt.close(fig)

    # Figure 2: per-variant robustness score bar chart
    robustness = analysis["robustness"]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2e7d32" if v >= 0.65 else "#c62828" for v in robustness.values]
    ax.barh([f"{vid}\n{manifest_by_id[vid]['name']}" for vid in robustness.index],
            robustness.values, color=colors)
    ax.axvline(summary["pass_threshold"], color="black", linestyle="--", linewidth=1,
               label=f"pass threshold ({summary['pass_threshold']})")
    ax.set_xlabel("Mean accuracy across all 89 conditions (robustness score)")
    ax.set_title("Per-Variant Robustness Score")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "robustness_by_variant.png"), dpi=130)
    plt.close(fig)

    # Figure 3: failure rate by condition category
    cats = list(analysis["failure_rate_by_category"].keys())
    rates = [analysis["failure_rate_by_category"][c] for c in cats]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cats, rates, color="#ad1457")
    ax.set_ylabel("Failure rate (fraction of variant x condition cells below threshold)")
    ax.set_title("Failure Rate by Condition Category")
    ax.set_ylim(0, 1)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "failure_rate_by_category.png"), dpi=130)
    plt.close(fig)

    print(f"Saved 3 figures to {FIG_DIR}")


def write_report(matrix, manifest, analysis, summary, conditions):
    manifest_by_id = {m["id"]: m for m in manifest}
    robustness = analysis["robustness"]
    lines = []
    lines.append("# Robustness Evaluation Report\n")
    lines.append(f"Total evaluation runs: {summary['n_total_runs']} "
                  f"({summary['n_variants']} variants x {summary['n_conditions']} conditions)\n")
    lines.append(f"Pass threshold: accuracy >= {summary['pass_threshold']}\n")
    lines.append(f"**Overall pass rate: {summary['overall_pass_rate']:.3f}** "
                 f"({summary['n_total_runs'] - summary['n_failures']} / "
                 f"{summary['n_total_runs']} cells passed)\n")

    lines.append("\n## Per-Variant Robustness Scores (mean accuracy across all 89 conditions)\n")
    lines.append("| Rank | Variant ID | Name | Family | Robustness Score |")
    lines.append("|---|---|---|---|---|")
    for rank, (vid, score) in enumerate(robustness.items(), start=1):
        m = manifest_by_id[vid]
        lines.append(f"| {rank} | {vid} | {m['name']} | {m['family']} | {score:.3f} |")

    lines.append("\n## Failure Rate by Condition Category\n")
    lines.append("| Category | Failure Rate (cells below threshold) |")
    lines.append("|---|---|")
    for cat, rate in analysis["failure_rate_by_category"].items():
        lines.append(f"| {cat} | {rate:.3f} |")

    lines.append("\n## Worst-Performing Variant/Condition Pairs\n")
    worst = analysis["failures_df"].sort_values("accuracy").head(15)
    lines.append("| Variant | Condition | Category | Accuracy |")
    lines.append("|---|---|---|---|")
    for _, row in worst.iterrows():
        lines.append(f"| {row['variant_name']} | {row['condition']} | "
                      f"{row['category']} | {row['accuracy']:.3f} |")

    lines.append("\n## Discovered Failure-Mode Patterns\n")
    lines.append("A pattern = a variant scoring below the pass threshold on "
                 ">=50% of conditions within one category.\n")
    for p in sorted(analysis["patterns"], key=lambda x: -x["fail_fraction"])[:15]:
        lines.append(f"- **{p['variant_name']}** ({p['variant']}) fails "
                      f"{p['fail_fraction']:.0%} of `{p['category']}` conditions.")

    lines.append("\n## Safety Guardrail Recommendations\n")
    lines.append("Derived programmatically from the patterns and best-per-category "
                 "results above (not hand-written):\n")
    for g in summary["guardrails"]:
        lines.append(f"- {g}")

    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {os.path.join(OUT_DIR, 'report.md')}")


def main():
    run_step("generate_eval_conditions.py")
    run_step("train_model_variants.py")
    run_step("run_evaluation_suite.py")

    matrix = pd.read_csv(os.path.join(OUT_DIR, "eval_matrix.csv"), index_col=0)
    with open(os.path.join(MODEL_DIR, "variants_manifest.pkl"), "rb") as f:
        manifest = pickle.load(f)
    with open(os.path.join(OUT_DIR, "failure_modes.pkl"), "rb") as f:
        analysis = pickle.load(f)
    with open(os.path.join(OUT_DIR, "summary_stats.pkl"), "rb") as f:
        summary = pickle.load(f)
    with open(os.path.join(DATA_DIR, "conditions.pkl"), "rb") as f:
        conditions = pickle.load(f)

    make_figures(matrix, manifest, analysis, summary)
    write_report(matrix, manifest, analysis, summary, conditions)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Overall pass rate: {summary['overall_pass_rate']:.3f}")
    print(f"Total runs: {summary['n_total_runs']}, failures: {summary['n_failures']}")
    print(f"Best variant: {analysis['robustness'].idxmax()} "
          f"({analysis['robustness'].max():.3f})")
    print(f"Worst variant: {analysis['robustness'].idxmin()} "
          f"({analysis['robustness'].min():.3f})")


if __name__ == "__main__":
    main()
