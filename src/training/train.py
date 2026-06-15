"""Train a LightGBM day-ahead AESO pool-price forecaster.

Runs identically on a laptop or as an Azure ML job. Logs params, metrics and
the model to MLflow (AML captures these automatically when run as a job), and
writes a self-contained model folder (model.pkl + climatology.csv +
metrics.json) that the managed online endpoint serves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make the shared `common` package importable whether run locally or in AML.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.features import (  # noqa: E402
    EXOG_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    add_calendar_features,
    build_feature_matrix,
    feature_columns,
)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def clipped_mape(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 5.0) -> float:
    denom = np.clip(np.abs(y_true), floor, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def build_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Average exogenous drivers by (month, hour) for use as serving defaults."""
    enriched = add_calendar_features(df)
    clim = (
        enriched.groupby(["month", "hour"])[EXOG_COLUMNS]
        .mean()
        .reset_index()
        .round(3)
    )
    return clim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV (file or folder)")
    parser.add_argument("--model_output", default="model_artifacts")
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=64)
    parser.add_argument("--register_metrics_only", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.is_dir():
        csvs = list(data_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {data_path}")
        data_path = csvs[0]

    df = pd.read_csv(data_path, parse_dates=[TIME_COLUMN]).sort_values(TIME_COLUMN)
    print(f"Loaded {len(df):,} rows spanning {df[TIME_COLUMN].min()} -> {df[TIME_COLUMN].max()}")

    # Time-based split: never shuffle a time series.
    split_idx = int(len(df) * (1 - args.test_fraction))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = build_feature_matrix(train_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_test = build_feature_matrix(test_df)
    y_test = test_df[TARGET_COLUMN].to_numpy()

    params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    mlflow.log_params(params)

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 999.99)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "smape": smape(y_test, preds),
        "mape_clipped": clipped_mape(y_test, preds),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    print("Evaluation metrics:", json.dumps(metrics, indent=2))

    # Feature importances -> useful for the LLM "what's driving price" answer.
    importances = dict(
        sorted(
            zip(feature_columns(), model.feature_importances_.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    out_dir = Path(args.model_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.pkl")
    build_climatology(df).to_csv(out_dir / "climatology.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "feature_importances.json").write_text(json.dumps(importances, indent=2))
    (out_dir / "model_card.json").write_text(
        json.dumps(
            {
                "name": "aeso-price-forecaster",
                "task": "Day-ahead hourly AESO pool price forecast (CAD/MWh)",
                "algorithm": "LightGBM regressor",
                "features": feature_columns(),
                "exogenous_drivers": EXOG_COLUMNS,
                "metrics": metrics,
                "top_features": list(importances.keys())[:8],
                "training_rows": int(len(df)),
                "offer_cap_cad_mwh": 999.99,
            },
            indent=2,
        )
    )

    # Also log an MLflow model flavor for lineage/registry richness.
    try:
        mlflow.sklearn.log_model(model, artifact_path="mlflow-model")
    except Exception as exc:  # pragma: no cover
        print(f"(non-fatal) mlflow model log skipped: {exc}")

    print(f"Wrote model artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
