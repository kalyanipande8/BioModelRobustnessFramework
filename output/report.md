# Robustness Evaluation Report

Total evaluation runs: 1602 (18 variants x 89 conditions)

Pass threshold: accuracy >= 0.65

**Overall pass rate: 0.758** (1215 / 1602 cells passed)


## Per-Variant Robustness Scores (mean accuracy across all 89 conditions)

| Rank | Variant ID | Name | Family | Robustness Score |
|---|---|---|---|---|
| 1 | variant_01 | logreg_C0.01 | LogisticRegression | 0.799 |
| 2 | variant_03 | logreg_C100.0 | LogisticRegression | 0.798 |
| 3 | variant_02 | logreg_C1.0 | LogisticRegression | 0.798 |
| 4 | variant_09 | mlp_32x16_alpha1.0 | MLPClassifier | 0.793 |
| 5 | variant_06 | mlp_32x16 | MLPClassifier | 0.790 |
| 6 | variant_04 | mlp_16 | MLPClassifier | 0.790 |
| 7 | variant_05 | mlp_64 | MLPClassifier | 0.789 |
| 8 | variant_08 | mlp_32x16_alpha0.01 | MLPClassifier | 0.787 |
| 9 | variant_07 | mlp_64x32x16 | MLPClassifier | 0.781 |
| 10 | variant_14 | extratrees_default | ExtraTreesClassifier | 0.775 |
| 11 | variant_13 | gboost_default | GradientBoostingClassifier | 0.774 |
| 12 | variant_16 | svm_rbf | SVC | 0.766 |
| 13 | variant_12 | rf_depthNone_n100 | RandomForestClassifier | 0.755 |
| 14 | variant_11 | rf_depth8_n50 | RandomForestClassifier | 0.754 |
| 15 | variant_18 | rf_bootstrap_subset50pct | RandomForestClassifier | 0.724 |
| 16 | variant_10 | rf_depth3_n20 | RandomForestClassifier | 0.724 |
| 17 | variant_15 | decisiontree_depth4 | DecisionTreeClassifier | 0.714 |
| 18 | variant_17 | mlp_32x16_subset25pct | MLPClassifier | 0.493 |

## Failure Rate by Condition Category

| Category | Failure Rate (cells below threshold) |
|---|---|
| adversarial | 0.058 |
| class_imbalance | 0.052 |
| ood_shift | 0.086 |
| noise | 0.611 |
| missing_feature | 0.401 |

## Worst-Performing Variant/Condition Pairs

| Variant | Condition | Category | Accuracy |
|---|---|---|---|
| mlp_32x16_subset25pct | imbalance_15_posratio_0.85 | class_imbalance | 0.221 |
| mlp_32x16_subset25pct | imbalance_14_posratio_0.79 | class_imbalance | 0.290 |
| mlp_32x16_subset25pct | imbalance_13_posratio_0.74 | class_imbalance | 0.334 |
| mlp_32x16_subset25pct | imbalance_12_posratio_0.68 | class_imbalance | 0.385 |
| mlp_32x16_subset25pct | imbalance_11_posratio_0.62 | class_imbalance | 0.425 |
| mlp_32x16_subset25pct | imbalance_10_posratio_0.56 | class_imbalance | 0.451 |
| mlp_32x16_subset25pct | ood_shift_16_mag_3.18 | ood_shift | 0.455 |
| mlp_32x16_subset25pct | ood_shift_19_mag_3.79 | ood_shift | 0.465 |
| mlp_32x16_subset25pct | ood_shift_15_mag_2.97 | ood_shift | 0.465 |
| mlp_32x16_subset25pct | ood_shift_18_mag_3.59 | ood_shift | 0.468 |
| mlp_32x16_subset25pct | ood_shift_17_mag_3.38 | ood_shift | 0.470 |
| mlp_32x16_subset25pct | noise_17_sigma_2.53 | noise | 0.472 |
| mlp_32x16_subset25pct | ood_shift_09_mag_1.74 | ood_shift | 0.472 |
| mlp_32x16_subset25pct | noise_16_sigma_2.38 | noise | 0.475 |
| mlp_32x16_subset25pct | noise_14_sigma_2.07 | noise | 0.477 |

## Discovered Failure-Mode Patterns

A pattern = a variant scoring below the pass threshold on >=50% of conditions within one category.

- **mlp_32x16_subset25pct** (variant_17) fails 100% of `adversarial` conditions.
- **mlp_32x16_subset25pct** (variant_17) fails 100% of `ood_shift` conditions.
- **mlp_32x16_subset25pct** (variant_17) fails 100% of `noise` conditions.
- **mlp_32x16_subset25pct** (variant_17) fails 100% of `missing_feature` conditions.
- **rf_depth3_n20** (variant_10) fails 80% of `noise` conditions.
- **decisiontree_depth4** (variant_15) fails 80% of `noise` conditions.
- **mlp_32x16_subset25pct** (variant_17) fails 73% of `class_imbalance` conditions.
- **decisiontree_depth4** (variant_15) fails 71% of `missing_feature` conditions.
- **rf_bootstrap_subset50pct** (variant_18) fails 65% of `noise` conditions.
- **rf_depth8_n50** (variant_11) fails 60% of `noise` conditions.
- **rf_depthNone_n100** (variant_12) fails 60% of `noise` conditions.
- **rf_depth3_n20** (variant_10) fails 57% of `missing_feature` conditions.
- **logreg_C0.01** (variant_01) fails 55% of `noise` conditions.
- **logreg_C1.0** (variant_02) fails 55% of `noise` conditions.
- **logreg_C100.0** (variant_03) fails 55% of `noise` conditions.

## Safety Guardrail Recommendations

Derived programmatically from the patterns and best-per-category results above (not hand-written):

- Do NOT deploy mlp_32x16_subset25pct (variant_17) in settings resembling 'adversarial' conditions — it fails (<65% accuracy) on 100% of adversarial test conditions.
- Do NOT deploy mlp_32x16_subset25pct (variant_17) in settings resembling 'ood_shift' conditions — it fails (<65% accuracy) on 100% of ood_shift test conditions.
- Do NOT deploy mlp_32x16_subset25pct (variant_17) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 100% of noise test conditions.
- Do NOT deploy mlp_32x16_subset25pct (variant_17) in settings resembling 'missing_feature' conditions — it fails (<65% accuracy) on 100% of missing_feature test conditions.
- Do NOT deploy rf_depth3_n20 (variant_10) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 80% of noise test conditions.
- Do NOT deploy decisiontree_depth4 (variant_15) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 80% of noise test conditions.
- Do NOT deploy mlp_32x16_subset25pct (variant_17) in settings resembling 'class_imbalance' conditions — it fails (<65% accuracy) on 73% of class_imbalance test conditions.
- Do NOT deploy decisiontree_depth4 (variant_15) in settings resembling 'missing_feature' conditions — it fails (<65% accuracy) on 71% of missing_feature test conditions.
- Do NOT deploy rf_bootstrap_subset50pct (variant_18) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 65% of noise test conditions.
- Do NOT deploy rf_depth8_n50 (variant_11) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 60% of noise test conditions.
- Do NOT deploy rf_depthNone_n100 (variant_12) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 60% of noise test conditions.
- Do NOT deploy rf_depth3_n20 (variant_10) in settings resembling 'missing_feature' conditions — it fails (<65% accuracy) on 57% of missing_feature test conditions.
- Do NOT deploy logreg_C0.01 (variant_01) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy logreg_C1.0 (variant_02) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy logreg_C100.0 (variant_03) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy mlp_64 (variant_05) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy mlp_32x16 (variant_06) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy mlp_64x32x16 (variant_07) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy mlp_32x16_alpha0.01 (variant_08) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy mlp_32x16_alpha1.0 (variant_09) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy gboost_default (variant_13) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy extratrees_default (variant_14) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy svm_rbf (variant_16) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 55% of noise test conditions.
- Do NOT deploy mlp_16 (variant_04) in settings resembling 'noise' conditions — it fails (<65% accuracy) on 50% of noise test conditions.
- Do NOT deploy rf_depth8_n50 (variant_11) in settings resembling 'missing_feature' conditions — it fails (<65% accuracy) on 50% of missing_feature test conditions.
- Do NOT deploy rf_bootstrap_subset50pct (variant_18) in settings resembling 'missing_feature' conditions — it fails (<65% accuracy) on 50% of missing_feature test conditions.
- For 'adversarial' conditions, prefer logreg_C0.01 (variant_01) — highest observed mean accuracy (0.988) among all 18 variants for this condition category.
- For 'class_imbalance' conditions, prefer logreg_C0.01 (variant_01) — highest observed mean accuracy (0.831) among all 18 variants for this condition category.
- For 'ood_shift' conditions, prefer logreg_C0.01 (variant_01) — highest observed mean accuracy (0.789) among all 18 variants for this condition category.
- For 'noise' conditions, prefer mlp_16 (variant_04) — highest observed mean accuracy (0.674) among all 18 variants for this condition category.
- For 'missing_feature' conditions, prefer logreg_C1.0 (variant_02) — highest observed mean accuracy (0.695) among all 18 variants for this condition category.
- Overall least-robust variant across all 89 conditions: variant_17 (mean accuracy 0.493). Do not use as a default/production model without additional mitigation.
