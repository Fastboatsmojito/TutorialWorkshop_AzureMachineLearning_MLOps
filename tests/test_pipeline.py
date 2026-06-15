"""Lightweight tests exercised by CI: features, training, and scoring."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))


def _tiny_df():
    from generate_dataset import generate

    return generate("2024-01-01", "2024-04-01")  # ~3 months


def test_calendar_features_present():
    from common.features import build_feature_matrix, feature_columns

    df = _tiny_df()
    X = build_feature_matrix(df)
    assert list(X.columns) == feature_columns()
    assert len(X) == len(df)
    assert not X.isnull().any().any()


def test_train_and_score(tmp_path):
    from common.features import TARGET_COLUMN  # noqa: F401

    df = _tiny_df()
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)

    # Train into a temp model dir.
    out = tmp_path / "model"
    sys.argv = [
        "train.py", "--data", str(csv), "--model_output", str(out),
        "--n_estimators", "60",
    ]
    import importlib

    import mlflow

    with mlflow.start_run():
        train = importlib.import_module("training.train")
        train.main()

    assert (out / "model.pkl").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["rmse"] > 0

    # Score using the produced artifacts.
    os.environ["AZUREML_MODEL_DIR"] = str(out)
    score = importlib.import_module("training.score")
    score.init()
    result = score.run(json.dumps({"horizon_hours": 12}))
    assert len(result["forecast"]) == 12
    assert result["units"] == "CAD/MWh"
    assert all(0 <= r["predicted_price"] <= 999.99 for r in result["forecast"])
