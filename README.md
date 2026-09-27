# Model Robustness Evaluation Framework for Biology

A real, runnable robustness/adversarial-testing suite that trains 18 distinct
small model variants on a synthetic biological classification task and
evaluates every variant against 89 distinct stress conditions (18 x 89 =
1,602 real evaluation runs), then mines the resulting performance matrix for
genuine failure modes and derives safety-guardrail recommendations from them.

This project implements the resume line:

> Model Robustness Evaluation Framework for Biology | Benchmarking,
> Adversarial Testing — Developed comprehensive evaluation suite testing 18
> model variants across 89 biological conditions; identified failure modes,
> documented safety guardrails enabling domain expert deployment confidence.

## What this actually is

This is **not** an evaluation of real large biological foundation models
(e.g. ESM, AlphaFold, a genomic LLM). It is a small, from-scratch,
laptop-CPU-scale implementation that genuinely exercises the same
methodology — train multiple model variants, define a battery of stress
conditions standing in for real biological deployment hazards, run every
variant against every condition, and mine the results for failure modes —
on a synthetic "protein-property classification" task built specifically
for this project.

- **The task**: a synthetic binary classification problem standing in for
  "does this protein sequence fold into a stable/functional structure?" —
  30 numeric features (10 informative, 20 redundant/noise), generated from a
  fixed logistic latent function with one nonlinear interaction term, fixed
  seed (42).
- **The 18 "model variants"**: small, genuinely-trained, from-scratch
  scikit-learn models (logistic regression, small MLPs, random forests,
  gradient boosting, extra-trees, a decision tree, and an SVM), varying
  architecture/depth/width, regularization strength, and training-data
  subset/resampling. See the variant table below — every variant was
  actually fit with `model.fit(...)` on real (synthetic) training data.
- **The 89 "biological conditions"**: synthetic perturbations/subsets of the
  held-out test set standing in for realistic evaluation hazards (see table
  below). All conditions are genuinely computed transformations of the same
  base test set — no numbers are hand-written.
- **The 1,602 evaluation results**: every one of the 18 x 89 cells is a real
  `model.predict(...)` call scored with `accuracy_score`, not a simulated or
  fabricated number.

### PyTorch vs. scikit-learn: which path was used, and why

Before starting, this machine was checked:

```
$ python3 -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'

$ df -h /
Filesystem   Size  Used  Avail  Capacity  Mounted on
/dev/disk3s3s1  228Gi  12Gi  980Mi    93%    /
```

PyTorch was not installed, and the root volume had only **980Mi free** —
not enough headroom to safely install a PyTorch CPU wheel (typically
100-250MB+ once dependencies resolve) without risking filling the disk
during an active build. Given the low-disk warning, this project uses the
**scikit-learn fallback path** exclusively: 18 variants across six sklearn
model families (`LogisticRegression`, `MLPClassifier`, `RandomForestClassifier`,
`GradientBoostingClassifier`, `ExtraTreesClassifier`, `DecisionTreeClassifier`,
`SVC`) instead of configurable PyTorch nets. This is documented honestly
here and in code comments — there is no PyTorch code path in this repo.

## The 18 model variants

All variants are trained by `train_model_variants.py` on a shared base
training set (`data/base_train.npz`, 2,000 samples), with a fixed seed
(42 + variant index) controlling any subsampling.

| ID | Name | Family | Config | Train subset |
|---|---|---|---|---|
| variant_01 | logreg_C0.01 | LogisticRegression | C=0.01, penalty=l2 | 100% |
| variant_02 | logreg_C1.0 | LogisticRegression | C=1.0, penalty=l2 | 100% |
| variant_03 | logreg_C100.0 | LogisticRegression | C=100.0, penalty=l2 | 100% |
| variant_04 | mlp_16 | MLPClassifier | hidden_layer_sizes=(16,), alpha=1e-4 | 100% |
| variant_05 | mlp_64 | MLPClassifier | hidden_layer_sizes=(64,), alpha=1e-4 | 100% |
| variant_06 | mlp_32x16 | MLPClassifier | hidden_layer_sizes=(32,16), alpha=1e-4 | 100% |
| variant_07 | mlp_64x32x16 | MLPClassifier | hidden_layer_sizes=(64,32,16), alpha=1e-4 | 100% |
| variant_08 | mlp_32x16_alpha0.01 | MLPClassifier | hidden_layer_sizes=(32,16), alpha=1e-2 | 100% |
| variant_09 | mlp_32x16_alpha1.0 | MLPClassifier | hidden_layer_sizes=(32,16), alpha=1.0 | 100% |
| variant_10 | rf_depth3_n20 | RandomForestClassifier | max_depth=3, n_estimators=20 | 100% |
| variant_11 | rf_depth8_n50 | RandomForestClassifier | max_depth=8, n_estimators=50 | 100% |
| variant_12 | rf_depthNone_n100 | RandomForestClassifier | max_depth=None, n_estimators=100 | 100% |
| variant_13 | gboost_default | GradientBoostingClassifier | n_estimators=100, max_depth=3, lr=0.1 | 100% |
| variant_14 | extratrees_default | ExtraTreesClassifier | n_estimators=100, max_depth=None | 100% |
| variant_15 | decisiontree_depth4 | DecisionTreeClassifier | max_depth=4 | 100% |
| variant_16 | svm_rbf | SVC | kernel=rbf, C=1.0, probability=True | 100% |
| variant_17 | mlp_32x16_subset25pct | MLPClassifier | hidden_layer_sizes=(32,16), alpha=1e-4 | 25% (no replace) |
| variant_18 | rf_bootstrap_subset50pct | RandomForestClassifier | max_depth=8, n_estimators=50 | 50% (bootstrap) |

## The 89 evaluation condition categories

Built by `generate_eval_conditions.py` from the same held-out base test set
(400 samples), using a single shared latent decision function so every
condition reflects the same underlying task:

| Category | Count | What it stands in for | How it's built |
|---|---|---|---|
| `noise` | 20 | Sequencing-error-rate variation | Additive Gaussian feature noise, sigma linearly swept 0.05 -> 3.00 |
| `ood_shift` | 20 | Novel organism/tissue distribution shift | Mean shift + variance inflation along a random unit direction, magnitude swept 0.10 -> 4.00 |
| `class_imbalance` | 15 | Rare-condition prevalence shift | Resampled test subsets with positive-class ratio swept 0.05 -> 0.85 |
| `adversarial` | 20 | Rare/edge-case sequence variants | Targeted perturbation of the 8 most class-discriminative features toward the opposite class, magnitude swept 0.10 -> 5.00 |
| `missing_feature` | 14 | Missing/low-coverage sequencing data | Increasing fraction (0.05 -> 0.95) of feature entries mean-imputed/masked |

20 + 20 + 15 + 20 + 14 = **89 conditions**.

## How to run

```bash
pip install -r requirements.txt
python3 run_all.py        # generate conditions -> train 18 variants -> run 1,602 evals -> figures + report
pytest -q                 # 7 tests, real (small/fast) end-to-end pipeline run
```

Individual stages can also be run directly: `generate_eval_conditions.py`,
`train_model_variants.py`, `run_evaluation_suite.py`.

## Results (from an actual run on this machine)

Full data: [`output/report.md`](output/report.md), [`output/eval_matrix.csv`](output/eval_matrix.csv) (regenerate via `python3 run_all.py`).

- **1,602 / 1,602 real evaluation runs completed** (18 variants x 89 conditions)
- **Overall pass rate (accuracy >= 0.65): 0.758** (1,215 / 1,602 cells passed)
- **Best variant overall**: `logreg_C0.01` (variant_01), robustness score 0.799
- **Worst variant overall**: `mlp_32x16_subset25pct` (variant_17), robustness score 0.493
- **Failure rate by condition category**: noise 0.611, missing_feature 0.401, ood_shift 0.086, adversarial 0.058, class_imbalance 0.052

### Concrete failure modes discovered (from the real 18x89 matrix)

1. **`mlp_32x16_subset25pct` (variant_17), trained on only 25% of the
   training data, catastrophically fails almost everywhere** — 100% failure
   on `adversarial`, `ood_shift`, `noise`, and `missing_feature` conditions,
   and 73% on `class_imbalance`. Its overall robustness score (0.493) is
   barely above chance, versus 0.72-0.80 for every other variant. This is a
   genuine training-data-starvation failure mode: the same architecture
   (`mlp_32x16`, alpha=1e-4) trained on 100% of the data scores 0.790.
2. **Shallow tree-based models are disproportionately fragile under noise**:
   `rf_depth3_n20` and `decisiontree_depth4` both fail on 80% of `noise`
   conditions (worst of any variant on that category besides variant_17),
   while deeper/larger ensembles (`rf_depthNone_n100`, `extratrees_default`,
   `gboost_default`) are more resilient. Shallow trees rely on a small
   number of axis-aligned splits, which are easily knocked past a decision
   boundary by additive feature noise.
3. **`noise` is by far the hardest condition category** (0.611 failure rate
   across all variants, vs. 0.052-0.086 for adversarial/imbalance/OOD
   conditions once magnitude is moderate). Every one of the 18 variants
   fails at least half of the 20 noise conditions except `mlp_16`, which is
   the single most noise-robust variant (mean accuracy 0.674 on noise
   conditions) — likely because its very small hidden layer (16 units)
   generalizes more smoothly and doesn't overfit to noise-sensitive feature
   interactions.
4. **Class imbalance and moderate adversarial perturbation are comparatively
   well-handled** by most variants (failure rates 0.052 and 0.058) — logistic
   regression variants in particular are highly robust to the adversarial
   perturbation used here (mean accuracy 0.988 for `logreg_C0.01`), because
   a linear decision boundary is less sensitive to perturbations targeted at
   a fixed set of "most discriminative" features than models that rely on
   sharper nonlinear splits.

### Guardrail recommendations (derived programmatically from the matrix above)

- **Do not deploy `mlp_32x16_subset25pct` (variant_17) in production at
  all** without retraining on the full dataset — it is the single
  least-robust variant across every condition category (overall robustness
  0.493), confirming that training-data volume is a first-order robustness
  factor for this task.
- **Do not deploy `rf_depth3_n20` or `decisiontree_depth4` in
  high-measurement-noise settings** (e.g. noisy sequencing pipelines) — both
  fail 80% of noise conditions.
- **For noise-heavy deployments, prefer `mlp_16`** (mean accuracy 0.674 on
  noise conditions, the best of all 18 variants for this category).
- **For novel-organism/tissue (OOD) and rare-variant (adversarial) inputs,
  prefer `logreg_C0.01`** (mean accuracy 0.789 on OOD-shift conditions and
  0.988 on adversarial conditions — the top performer in both categories).
- **For missing/low-coverage feature data, prefer `logreg_C1.0`** (mean
  accuracy 0.695 on missing-feature conditions, the best of all 18
  variants).

## Figures

Generated to `output/figures/` by `run_all.py`:

- `variant_condition_heatmap.png` — full 18 x 89 accuracy matrix
- `robustness_by_variant.png` — per-variant robustness score bar chart, with pass threshold marked
- `failure_rate_by_category.png` — failure rate by condition category

## Project structure

```
BioModelRobustnessFramework/
├── generate_eval_conditions.py   # builds base dataset + 89 conditions
├── train_model_variants.py       # trains 18 model variants
├── run_evaluation_suite.py       # runs 18x89 real evals, finds failure modes, derives guardrails
├── run_all.py                    # orchestrates the full pipeline + figures + report.md
├── tests/
│   └── test_pipeline.py          # real pytest tests over a fast pipeline run
├── data/                         # generated (gitignored): base dataset + conditions
├── models/                       # generated (gitignored): 18 trained model .pkl files
├── output/
│   ├── report.md                 # full results report (committed)
│   ├── eval_matrix.csv           # generated (gitignored): 18x89 accuracy matrix
│   └── figures/                  # 3 committed PNG figures
├── requirements.txt
└── README.md
```

## Tech stack

Python 3, scikit-learn (LogisticRegression, MLPClassifier, RandomForest,
GradientBoosting, ExtraTrees, DecisionTree, SVC), NumPy, pandas, matplotlib,
pytest. No PyTorch (see "PyTorch vs. scikit-learn" above for why).
