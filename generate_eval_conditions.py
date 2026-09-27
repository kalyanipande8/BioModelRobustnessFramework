"""
generate_eval_conditions.py

Generates a base synthetic biological prediction task (protein-property
classification: predict whether a synthetic protein sequence-derived feature
vector corresponds to a "stable/functional" fold class vs "unstable/non-
functional" class), then builds 89 distinct evaluation CONDITIONS by applying
perturbations/subsets to that base evaluation set.

Everything is synthetic / from-scratch. This is a stand-in task, not real
biological data. Fixed random seed for reproducibility.

Condition categories (89 total):
  1. Noise conditions (sequencing-error-rate stand-in): 20 conditions
     Gaussian feature noise at increasing sigma levels.
  2. OOD feature-shift conditions (novel organism/tissue stand-in): 20
     conditions. Mean/covariance shift applied to features.
  3. Class-imbalance conditions: 15 conditions. Resampled subsets with
     varying positive:negative ratios.
  4. Adversarial perturbation conditions (rare-variant / edge-case
     stand-in): 20 conditions. Small targeted perturbations along the
     model-agnostic gradient-free direction (sign-flip on top-k
     discriminative features) at increasing magnitude.
  5. Missing-feature conditions: 14 conditions. Randomly zero out an
     increasing fraction of features (mean-imputation-style masking).

Total: 20 + 20 + 15 + 20 + 14 = 89

Output: writes data/base_train.npz, data/base_test.npz, and
data/conditions.pkl (list of dicts: name, category, X, y, description).
"""
import numpy as np
import os
import pickle

SEED = 42
N_FEATURES = 30
N_BASE_TRAIN = 2000
N_BASE_TEST = 400

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def make_base_dataset(rng, n_samples, w, perm, n_features=N_FEATURES):
    """Synthetic 'protein property classification' task.

    Features stand in for sequence-derived descriptors (e.g. hydrophobicity
    index, charge, length-normalized composition stats, secondary-structure
    propensities). Label = 1 means 'stable/functional fold', 0 means
    'unstable/non-functional'.

    IMPORTANT: `w` (the latent informative direction) and `perm` (the column
    shuffle) must be generated ONCE and shared across every call (train,
    test, and all condition variants) so that every split reflects the same
    underlying decision boundary. Generating fresh w/perm per call would
    silently make train and test come from unrelated tasks.
    """
    n_informative = len(w)
    X_informative = rng.normal(0, 1, size=(n_samples, n_informative))
    logits = X_informative @ w
    # nonlinear interaction term to make the task non-trivial
    logits += 0.5 * X_informative[:, 0] * X_informative[:, 1]
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)

    n_redundant = n_features - n_informative
    X_redundant = rng.normal(0, 1, size=(n_samples, n_redundant))
    X = np.concatenate([X_informative, X_redundant], axis=1)

    # shuffle column order (shared perm) so "informative" features aren't
    # trivially at front, but consistently placed across all splits
    X = X[:, perm]
    return X.astype(np.float32), y.astype(int)


def top_k_discriminative_features(X, y, k=8):
    """Rank features by |mean difference between classes| / pooled std."""
    mu1 = X[y == 1].mean(axis=0)
    mu0 = X[y == 0].mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    score = np.abs(mu1 - mu0) / sd
    return np.argsort(score)[::-1][:k]


def build_conditions(rng, X_test, y_test):
    conditions = []

    # ---- Category 1: Noise conditions (20) — sequencing-error-rate stand-in
    sigmas = np.linspace(0.05, 3.0, 20)
    for i, sigma in enumerate(sigmas):
        noise = rng.normal(0, sigma, size=X_test.shape)
        conditions.append({
            "name": f"noise_{i+1:02d}_sigma_{sigma:.2f}",
            "category": "noise",
            "X": (X_test + noise).astype(np.float32),
            "y": y_test.copy(),
            "description": f"Gaussian feature noise sigma={sigma:.2f} "
                            f"(stand-in for sequencing-error rate).",
        })

    # ---- Category 2: OOD feature-shift conditions (20) — novel organism/tissue
    shifts = np.linspace(0.1, 4.0, 20)
    for i, shift in enumerate(shifts):
        direction = rng.normal(0, 1, size=X_test.shape[1])
        direction /= np.linalg.norm(direction)
        scale = 1.0 + (shift / 4.0)  # also inflate variance a bit
        X_shifted = (X_test + shift * direction) * scale
        conditions.append({
            "name": f"ood_shift_{i+1:02d}_mag_{shift:.2f}",
            "category": "ood_shift",
            "X": X_shifted.astype(np.float32),
            "y": y_test.copy(),
            "description": f"Mean/variance feature shift magnitude={shift:.2f} "
                            f"(stand-in for novel organism/tissue distribution).",
        })

    # ---- Category 3: Class-imbalance conditions (15)
    pos_ratios = np.linspace(0.05, 0.85, 15)
    idx_pos = np.where(y_test == 1)[0]
    idx_neg = np.where(y_test == 0)[0]
    for i, ratio in enumerate(pos_ratios):
        n_total = min(len(idx_pos), len(idx_neg)) * 2
        n_pos = int(n_total * ratio)
        n_neg = n_total - n_pos
        n_pos = min(n_pos, len(idx_pos))
        n_neg = min(n_neg, len(idx_neg))
        sel_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sel_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(sel)
        conditions.append({
            "name": f"imbalance_{i+1:02d}_posratio_{ratio:.2f}",
            "category": "class_imbalance",
            "X": X_test[sel].astype(np.float32),
            "y": y_test[sel].copy(),
            "description": f"Resampled test subset with positive-class ratio="
                            f"{ratio:.2f}.",
        })

    # ---- Category 4: Adversarial perturbation conditions (20) — rare-variant/edge-case
    top_feats = top_k_discriminative_features(X_test, y_test, k=8)
    magnitudes = np.linspace(0.1, 5.0, 20)
    for i, mag in enumerate(magnitudes):
        X_adv = X_test.copy()
        # flip sign / push top discriminative features toward the *wrong*
        # class boundary by magnitude `mag` (an untargeted, model-agnostic
        # adversarial-style perturbation on the most label-informative
        # features, standing in for rare/edge-case sequence variants).
        sign = np.where(y_test == 1, -1, 1).reshape(-1, 1)
        X_adv[:, top_feats] = X_adv[:, top_feats] + sign * mag
        conditions.append({
            "name": f"adversarial_{i+1:02d}_mag_{mag:.2f}",
            "category": "adversarial",
            "X": X_adv.astype(np.float32),
            "y": y_test.copy(),
            "description": f"Targeted perturbation of top-8 discriminative "
                            f"features toward opposite class, magnitude={mag:.2f} "
                            f"(stand-in for rare/edge-case variants).",
        })

    # ---- Category 5: Missing-feature conditions (14)
    missing_fracs = np.linspace(0.05, 0.95, 14)
    col_means = X_test.mean(axis=0)
    for i, frac in enumerate(missing_fracs):
        X_missing = X_test.copy()
        mask = rng.uniform(0, 1, size=X_missing.shape) < frac
        X_missing[mask] = np.take(col_means, np.where(mask)[1])
        conditions.append({
            "name": f"missing_{i+1:02d}_frac_{frac:.2f}",
            "category": "missing_feature",
            "X": X_missing.astype(np.float32),
            "y": y_test.copy(),
            "description": f"Fraction={frac:.2f} of feature entries mean-"
                            f"imputed/masked (missing sequencing coverage stand-in).",
        })

    assert len(conditions) == 89, f"expected 89 conditions, got {len(conditions)}"
    return conditions


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    n_informative = 10
    w = rng.normal(0, 1, size=n_informative)
    perm = rng.permutation(N_FEATURES)

    X_train, y_train = make_base_dataset(rng, N_BASE_TRAIN, w, perm)
    X_test, y_test = make_base_dataset(rng, N_BASE_TEST, w, perm)

    np.savez(os.path.join(OUT_DIR, "base_train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(OUT_DIR, "base_test.npz"), X=X_test, y=y_test)

    conditions = build_conditions(rng, X_test, y_test)
    with open(os.path.join(OUT_DIR, "conditions.pkl"), "wb") as f:
        pickle.dump(conditions, f)

    cat_counts = {}
    for c in conditions:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1

    print(f"Base train: {X_train.shape}, base test: {X_test.shape}")
    print(f"Generated {len(conditions)} conditions across categories:")
    for cat, n in cat_counts.items():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
