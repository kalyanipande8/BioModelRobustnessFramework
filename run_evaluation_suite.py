"""
run_evaluation_suite.py

Evaluates all 18 trained model variants against all 89 conditions
(18 x 89 = 1,602 real evaluation runs — real inference + real scoring,
not simulated numbers).

Builds a variant x condition accuracy matrix, computes per-variant
robustness scores, identifies genuine failure modes (variant/condition
pairs scoring below PASS_THRESHOLD), looks for patterns (e.g. a variant
failing consistently within one condition category), and derives safety
guardrail recommendations that logically follow from the discovered
failure modes.

Output: output/eval_matrix.csv, output/failure_modes.pkl,
output/summary_stats.pkl
"""
import numpy as np
import pandas as pd
import os
import pickle

from sklearn.metrics import accuracy_score

PASS_THRESHOLD = 0.65  # below this accuracy on a condition counts as a "failure"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "output")


def load_conditions():
    with open(os.path.join(DATA_DIR, "conditions.pkl"), "rb") as f:
        return pickle.load(f)


def load_manifest():
    with open(os.path.join(MODEL_DIR, "variants_manifest.pkl"), "rb") as f:
        return pickle.load(f)


def load_model(fname):
    with open(os.path.join(MODEL_DIR, fname), "rb") as f:
        return pickle.load(f)


def run_matrix(manifest, conditions):
    variant_ids = [m["id"] for m in manifest]
    condition_names = [c["name"] for c in conditions]
    matrix = pd.DataFrame(index=variant_ids, columns=condition_names, dtype=float)
    category_of = {c["name"]: c["category"] for c in conditions}

    for m in manifest:
        model = load_model(m["file"])
        for c in conditions:
            preds = model.predict(c["X"])
            acc = accuracy_score(c["y"], preds)
            matrix.loc[m["id"], c["name"]] = acc

    return matrix, category_of


def analyze_failure_modes(matrix, manifest, category_of):
    manifest_by_id = {m["id"]: m for m in manifest}
    failures = []  # list of dicts: variant, condition, category, accuracy
    for vid in matrix.index:
        for cond in matrix.columns:
            acc = matrix.loc[vid, cond]
            if acc < PASS_THRESHOLD:
                failures.append({
                    "variant": vid,
                    "variant_name": manifest_by_id[vid]["name"],
                    "condition": cond,
                    "category": category_of[cond],
                    "accuracy": acc,
                })
    failures_df = pd.DataFrame(failures)

    # Per-variant robustness score = mean accuracy across all 89 conditions
    robustness = matrix.mean(axis=1).sort_values(ascending=False)

    # Per-condition-category failure rate
    cat_series = pd.Series(category_of)
    cond_to_cat = cat_series
    total_by_cat = pd.Series({cat: sum(1 for c in category_of.values() if c == cat)
                               for cat in set(category_of.values())})
    if not failures_df.empty:
        fail_counts_by_cat = failures_df.groupby("category").size()
    else:
        fail_counts_by_cat = pd.Series(dtype=int)
    n_variants = len(matrix.index)
    failure_rate_by_category = {}
    for cat, n_cond in total_by_cat.items():
        n_fail = fail_counts_by_cat.get(cat, 0)
        failure_rate_by_category[cat] = n_fail / (n_cond * n_variants)

    # Discover per-variant "consistent failure in a category" patterns:
    # a variant fails in >= 50% of conditions within some category.
    patterns = []
    if not failures_df.empty:
        for vid in matrix.index:
            vfail = failures_df[failures_df["variant"] == vid]
            for cat in set(category_of.values()):
                n_cond_in_cat = total_by_cat[cat]
                n_fail_in_cat = (vfail["category"] == cat).sum()
                if n_cond_in_cat > 0 and n_fail_in_cat / n_cond_in_cat >= 0.5:
                    patterns.append({
                        "variant": vid,
                        "variant_name": manifest_by_id[vid]["name"],
                        "category": cat,
                        "fail_fraction": n_fail_in_cat / n_cond_in_cat,
                    })

    # Best variant per category (highest mean accuracy within category) —
    # used for guardrail "recommend variant Z for condition type Y".
    best_per_category = {}
    for cat in set(category_of.values()):
        cols = [c for c in matrix.columns if category_of[c] == cat]
        means = matrix[cols].mean(axis=1)
        best_vid = means.idxmax()
        best_per_category[cat] = {
            "variant": best_vid,
            "variant_name": manifest_by_id[best_vid]["name"],
            "mean_accuracy": means.max(),
        }

    return {
        "failures_df": failures_df,
        "robustness": robustness,
        "failure_rate_by_category": failure_rate_by_category,
        "patterns": patterns,
        "best_per_category": best_per_category,
    }


def build_guardrails(analysis, manifest):
    """Derive plain-English guardrail recommendations from the discovered
    failure patterns. These are generated programmatically from the actual
    matrix, not hand-written."""
    guardrails = []
    for p in sorted(analysis["patterns"], key=lambda x: -x["fail_fraction"]):
        guardrails.append(
            f"Do NOT deploy {p['variant_name']} ({p['variant']}) in settings "
            f"resembling '{p['category']}' conditions — it fails "
            f"(<{PASS_THRESHOLD:.0%} accuracy) on {p['fail_fraction']:.0%} of "
            f"{p['category']} test conditions."
        )
    for cat, best in analysis["best_per_category"].items():
        guardrails.append(
            f"For '{cat}' conditions, prefer {best['variant_name']} "
            f"({best['variant']}) — highest observed mean accuracy "
            f"({best['mean_accuracy']:.3f}) among all 18 variants for this "
            f"condition category."
        )
    worst_overall = analysis["robustness"].idxmin()
    guardrails.append(
        f"Overall least-robust variant across all 89 conditions: "
        f"{worst_overall} (mean accuracy "
        f"{analysis['robustness'][worst_overall]:.3f}). Do not use as a "
        f"default/production model without additional mitigation."
    )
    return guardrails


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = load_manifest()
    conditions = load_conditions()

    print(f"Running evaluation suite: {len(manifest)} variants x "
          f"{len(conditions)} conditions = {len(manifest) * len(conditions)} runs")

    matrix, category_of = run_matrix(manifest, conditions)
    matrix.to_csv(os.path.join(OUT_DIR, "eval_matrix.csv"))

    analysis = analyze_failure_modes(matrix, manifest, category_of)
    guardrails = build_guardrails(analysis, manifest)

    overall_pass_rate = (matrix >= PASS_THRESHOLD).values.mean()

    summary_stats = {
        "overall_pass_rate": overall_pass_rate,
        "pass_threshold": PASS_THRESHOLD,
        "n_variants": len(manifest),
        "n_conditions": len(conditions),
        "n_total_runs": len(manifest) * len(conditions),
        "n_failures": len(analysis["failures_df"]),
        "robustness_scores": analysis["robustness"].to_dict(),
        "failure_rate_by_category": analysis["failure_rate_by_category"],
        "guardrails": guardrails,
    }

    with open(os.path.join(OUT_DIR, "summary_stats.pkl"), "wb") as f:
        pickle.dump(summary_stats, f)
    with open(os.path.join(OUT_DIR, "failure_modes.pkl"), "wb") as f:
        pickle.dump(analysis, f)

    print(f"\nOverall pass rate (acc >= {PASS_THRESHOLD}): {overall_pass_rate:.3f}")
    print(f"Total failure cells: {len(analysis['failures_df'])} / "
          f"{len(manifest) * len(conditions)}")
    print("\nPer-variant robustness (mean accuracy across 89 conditions), best->worst:")
    for vid, score in analysis["robustness"].items():
        print(f"  {vid}: {score:.3f}")
    print("\nFailure rate by condition category:")
    for cat, rate in analysis["failure_rate_by_category"].items():
        print(f"  {cat}: {rate:.3f}")
    print("\nDiscovered patterns (variant consistently failing in a category):")
    for p in analysis["patterns"]:
        print(f"  {p['variant_name']} ({p['variant']}) fails {p['fail_fraction']:.0%} "
              f"of '{p['category']}' conditions")
    print("\nGuardrail recommendations:")
    for g in guardrails:
        print(f"  - {g}")


if __name__ == "__main__":
    main()
