"""
train_model_variants.py

Trains 18 distinct, genuinely different, small "model variants" on a shared
base training set (data/base_train.npz), varying:
  - model family (logistic regression, small MLP, random forest, gradient
    boosting, extra-trees, SVM)
  - architecture width/depth (for MLPs)
  - regularization strength
  - training-data subset size / bootstrap resampling

All models are small, from-scratch (scikit-learn) stand-ins — NOT real
biological foundation models. Fixed random seed for reproducibility.

Output: models/variant_<NN>_<name>.pkl plus models/variants_manifest.pkl
(list of dicts: id, name, family, config, train_subset_frac, train_accuracy).
"""
import numpy as np
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def subset(X, y, frac, seed, bootstrap=False):
    rng = np.random.default_rng(seed)
    n = len(y)
    k = max(20, int(n * frac))
    if bootstrap:
        idx = rng.choice(n, size=k, replace=True)
    else:
        idx = rng.choice(n, size=k, replace=False)
    return X[idx], y[idx]


def build_variant_specs():
    """Returns a list of 18 variant specs: (name, family, build_fn, train_frac, bootstrap)."""
    specs = []

    # 1-3: Logistic regression, varying regularization strength (C)
    for i, C in enumerate([0.01, 1.0, 100.0]):
        specs.append({
            "name": f"logreg_C{C}",
            "family": "LogisticRegression",
            "config": f"C={C}, penalty=l2, max_iter=1000",
            "build": lambda C=C: LogisticRegression(C=C, max_iter=1000, random_state=SEED),
            "train_frac": 1.0,
            "bootstrap": False,
        })

    # 4-7: Small MLPs, varying depth/width
    mlp_archs = [(16,), (64,), (32, 16), (64, 32, 16)]
    for arch in mlp_archs:
        specs.append({
            "name": f"mlp_{'x'.join(map(str, arch))}",
            "family": "MLPClassifier",
            "config": f"hidden_layer_sizes={arch}, alpha=1e-4, max_iter=800",
            "build": lambda arch=arch: MLPClassifier(
                hidden_layer_sizes=arch, alpha=1e-4, max_iter=800,
                random_state=SEED, early_stopping=True),
            "train_frac": 1.0,
            "bootstrap": False,
        })

    # 8-9: MLP varying regularization (alpha) at fixed architecture
    for alpha in [1e-2, 1e0]:
        specs.append({
            "name": f"mlp_32x16_alpha{alpha}",
            "family": "MLPClassifier",
            "config": f"hidden_layer_sizes=(32,16), alpha={alpha}, max_iter=800",
            "build": lambda alpha=alpha: MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=alpha, max_iter=800,
                random_state=SEED, early_stopping=True),
            "train_frac": 1.0,
            "bootstrap": False,
        })

    # 10-12: Random forest, varying max_depth / n_estimators
    rf_configs = [(3, 20), (8, 50), (None, 100)]
    for depth, n_est in rf_configs:
        specs.append({
            "name": f"rf_depth{depth}_n{n_est}",
            "family": "RandomForestClassifier",
            "config": f"max_depth={depth}, n_estimators={n_est}",
            "build": lambda depth=depth, n_est=n_est: RandomForestClassifier(
                max_depth=depth, n_estimators=n_est, random_state=SEED),
            "train_frac": 1.0,
            "bootstrap": False,
        })

    # 13: Gradient boosting
    specs.append({
        "name": "gboost_default",
        "family": "GradientBoostingClassifier",
        "config": "n_estimators=100, max_depth=3, learning_rate=0.1",
        "build": lambda: GradientBoostingClassifier(random_state=SEED),
        "train_frac": 1.0,
        "bootstrap": False,
    })

    # 14: Extra-trees
    specs.append({
        "name": "extratrees_default",
        "family": "ExtraTreesClassifier",
        "config": "n_estimators=100, max_depth=None",
        "build": lambda: ExtraTreesClassifier(n_estimators=100, random_state=SEED),
        "train_frac": 1.0,
        "bootstrap": False,
    })

    # 15: Single decision tree (shallow, likely to be brittle -> interesting for failure modes)
    specs.append({
        "name": "decisiontree_depth4",
        "family": "DecisionTreeClassifier",
        "config": "max_depth=4",
        "build": lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "train_frac": 1.0,
        "bootstrap": False,
    })

    # 16: SVM (RBF kernel)
    specs.append({
        "name": "svm_rbf",
        "family": "SVC",
        "config": "kernel=rbf, C=1.0, probability=True",
        "build": lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
        "train_frac": 1.0,
        "bootstrap": False,
    })

    # 17-18: Training-data-subset variants (same MLP architecture, different data)
    specs.append({
        "name": "mlp_32x16_subset25pct",
        "family": "MLPClassifier",
        "config": "hidden_layer_sizes=(32,16), alpha=1e-4, train_subset=25% (no replace)",
        "build": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-4, max_iter=800,
            random_state=SEED, early_stopping=True),
        "train_frac": 0.25,
        "bootstrap": False,
    })
    specs.append({
        "name": "rf_bootstrap_subset50pct",
        "family": "RandomForestClassifier",
        "config": "max_depth=8, n_estimators=50, train_subset=50% (bootstrap resample)",
        "build": lambda: RandomForestClassifier(max_depth=8, n_estimators=50, random_state=SEED),
        "train_frac": 0.5,
        "bootstrap": True,
    })

    assert len(specs) == 18, f"expected 18 variants, got {len(specs)}"
    return specs


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    data = np.load(os.path.join(DATA_DIR, "base_train.npz"))
    X_train, y_train = data["X"], data["y"]

    specs = build_variant_specs()
    manifest = []

    for i, spec in enumerate(specs):
        vid = f"variant_{i+1:02d}"
        Xs, ys = subset(X_train, y_train, spec["train_frac"], seed=SEED + i,
                         bootstrap=spec["bootstrap"])
        model = spec["build"]()
        model.fit(Xs, ys)
        train_acc = model.score(Xs, ys)

        fname = os.path.join(MODEL_DIR, f"{vid}_{spec['name']}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(model, f)

        manifest.append({
            "id": vid,
            "name": spec["name"],
            "family": spec["family"],
            "config": spec["config"],
            "train_subset_frac": spec["train_frac"],
            "bootstrap": spec["bootstrap"],
            "train_accuracy": train_acc,
            "file": os.path.basename(fname),
        })
        print(f"[{vid}] {spec['name']:35s} family={spec['family']:28s} "
              f"train_acc={train_acc:.3f}")

    with open(os.path.join(MODEL_DIR, "variants_manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)

    print(f"\nTrained {len(manifest)} model variants -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
