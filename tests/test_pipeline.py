"""
tests/test_pipeline.py

Real pytest tests that run a fast, small version of the full pipeline
(smaller train/test sizes, fewer trees etc. is not needed here -- the base
pipeline is already laptop-CPU-fast, so tests run the real pipeline against
a temp directory) and assert genuine properties of the output.
"""
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def pipeline_run():
    """Run generate -> train -> evaluate in an isolated temp copy of the repo
    so tests don't depend on / clobber the checked-in output/ directory."""
    tmpdir = tempfile.mkdtemp(prefix="biorobust_test_")
    try:
        for fname in ["generate_eval_conditions.py", "train_model_variants.py",
                      "run_evaluation_suite.py"]:
            shutil.copy(os.path.join(REPO_ROOT, fname), tmpdir)

        env = os.environ.copy()
        for script in ["generate_eval_conditions.py", "train_model_variants.py",
                       "run_evaluation_suite.py"]:
            result = subprocess.run(
                [sys.executable, script], cwd=tmpdir, env=env,
                capture_output=True, text=True,
            )
            assert result.returncode == 0, (
                f"{script} failed:\nstdout={result.stdout}\nstderr={result.stderr}"
            )

        with open(os.path.join(tmpdir, "data", "conditions.pkl"), "rb") as f:
            conditions = pickle.load(f)
        with open(os.path.join(tmpdir, "models", "variants_manifest.pkl"), "rb") as f:
            manifest = pickle.load(f)
        matrix = pd.read_csv(os.path.join(tmpdir, "output", "eval_matrix.csv"), index_col=0)
        with open(os.path.join(tmpdir, "output", "summary_stats.pkl"), "rb") as f:
            summary = pickle.load(f)
        with open(os.path.join(tmpdir, "output", "failure_modes.pkl"), "rb") as f:
            analysis = pickle.load(f)

        yield {
            "tmpdir": tmpdir,
            "conditions": conditions,
            "manifest": manifest,
            "matrix": matrix,
            "summary": summary,
            "analysis": analysis,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_89_conditions_generated(pipeline_run):
    assert len(pipeline_run["conditions"]) == 89
    names = [c["name"] for c in pipeline_run["conditions"]]
    assert len(set(names)) == 89, "condition names must be unique"


def test_18_variants_trained(pipeline_run):
    manifest = pipeline_run["manifest"]
    assert len(manifest) == 18
    ids = [m["id"] for m in manifest]
    assert len(set(ids)) == 18


def test_evaluation_matrix_complete_no_missing_entries(pipeline_run):
    matrix = pipeline_run["matrix"]
    assert matrix.shape == (18, 89), f"expected (18, 89), got {matrix.shape}"
    assert not matrix.isnull().values.any(), "evaluation matrix has missing entries"
    assert ((matrix.values >= 0) & (matrix.values <= 1)).all(), \
        "accuracy values must be in [0, 1]"


def test_at_least_one_genuine_failure_mode_found(pipeline_run):
    """Sanity check: the evaluation suite must actually find real weaknesses.
    If every cell passes, the conditions aren't stressing the models and the
    'failure mode discovery' claim would be meaningless."""
    summary = pipeline_run["summary"]
    assert summary["n_failures"] >= 1, (
        "No variant/condition pair scored below the pass threshold -- "
        "conditions are not adequately stressing the models."
    )
    failures_df = pipeline_run["analysis"]["failures_df"]
    assert len(failures_df) == summary["n_failures"]
    assert (failures_df["accuracy"] < summary["pass_threshold"]).all()


def test_robustness_scores_present_for_all_variants(pipeline_run):
    robustness = pipeline_run["analysis"]["robustness"]
    manifest_ids = {m["id"] for m in pipeline_run["manifest"]}
    assert set(robustness.index) == manifest_ids


def test_base_train_test_share_same_underlying_task(pipeline_run):
    """Regression test for a real bug found during development: train/test
    must be generated from the SAME latent decision boundary, otherwise a
    correctly-trained model would score near chance on clean test data."""
    tmpdir = pipeline_run["tmpdir"]
    train = np.load(os.path.join(tmpdir, "data", "base_train.npz"))
    test = np.load(os.path.join(tmpdir, "data", "base_test.npz"))
    manifest = pipeline_run["manifest"]

    # load one trained model and check it scores well above chance on the
    # *clean* (unperturbed) base test set
    model_file = os.path.join(tmpdir, "models", manifest[0]["file"])
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    acc = model.score(test["X"], test["y"])
    assert acc > 0.7, (
        f"variant scored only {acc:.3f} on clean base test set -- "
        f"train/test likely generated from mismatched underlying tasks"
    )


def test_guardrails_generated(pipeline_run):
    guardrails = pipeline_run["summary"]["guardrails"]
    assert len(guardrails) > 0
    assert all(isinstance(g, str) and len(g) > 0 for g in guardrails)
