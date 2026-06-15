"""Build a Responsible AI dashboard for the AESO price forecaster.

This is the "Explainability / RAI" pillar of the demo. It produces an Azure ML
**Responsible AI dashboard** (global + local feature explanations and error
analysis) that shows up in AML Studio under the registered model's
*Responsible AI* tab.

The RAI components require two things the serving pipeline doesn't:
  1. an **MLflow-flavored** model (the serving model is a custom folder), and
  2. **MLTable** train/test datasets whose columns are exactly the model's
     feature matrix + the target column.

So this script prepares both locally, registers them, then wires the official
Responsible AI components from the public `azureml` registry into a pipeline:

    constructor -> (explanation + error analysis) -> gather

Usage:
    python src/pipeline/rai_dashboard.py            # prep assets, submit, stream
    python src/pipeline/rai_dashboard.py --no-stream
    python src/pipeline/rai_dashboard.py --skip-prep   # reuse existing assets
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add src/
from common.features import TARGET_COLUMN, TIME_COLUMN, build_feature_matrix  # noqa: E402
from common.workspace import get_ml_client, load_resources  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "aeso_hourly.csv"

# Public Azure ML registry that hosts the Responsible AI components.
RAI_REGISTRY = "azureml"
RAI_TRAIN_DATASET = "aeso-rai-train"
RAI_TEST_DATASET = "aeso-rai-test"
RAI_MODEL_NAME = "aeso-price-forecaster-rai"

MLTABLE_YAML = """type: mltable
paths:
  - file: ./data.parquet
transformations:
  - read_parquet
"""


def _write_mltable(df: pd.DataFrame, folder: Path) -> Path:
    """Write a dataframe as an MLTable folder (data.parquet + MLTable yaml)."""
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / "data.parquet", index=False)
    (folder / "MLTable").write_text(MLTABLE_YAML)
    return folder


def prepare_assets(ml_client, test_fraction: float, csv_path: Path) -> dict:
    """Build feature+target MLTable datasets and an MLflow model, register all 3.

    Returns a dict of name:version references the RAI pipeline consumes.
    """
    import joblib  # noqa: F401  (kept for parity with training env)
    import mlflow.sklearn
    from lightgbm import LGBMRegressor
    from mlflow.models import infer_signature

    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data, Model

    df = pd.read_csv(csv_path, parse_dates=[TIME_COLUMN]).sort_values(TIME_COLUMN)
    # Mirror prep.py cleaning so RAI sees the same data the model trains on.
    df = df.drop_duplicates(subset=[TIME_COLUMN])
    df = df[(df[TARGET_COLUMN] >= 0) & (df[TARGET_COLUMN] <= 999.99)].dropna(subset=[TARGET_COLUMN])

    # RAI datasets = exactly the model's feature matrix + the target column.
    X = build_feature_matrix(df)
    y = df[TARGET_COLUMN].reset_index(drop=True)
    full = X.reset_index(drop=True).copy()
    full[TARGET_COLUMN] = y

    split_idx = int(len(full) * (1 - test_fraction))  # time-ordered split
    train_df, test_df = full.iloc[:split_idx], full.iloc[split_idx:]
    print(f"RAI datasets: train={len(train_df):,} rows, test={len(test_df):,} rows")

    feature_cols = list(X.columns)
    model = LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(train_df[feature_cols], train_df[TARGET_COLUMN])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_dir = _write_mltable(train_df, tmp_path / "train")
        test_dir = _write_mltable(test_df, tmp_path / "test")

        train_asset = ml_client.data.create_or_update(
            Data(name=RAI_TRAIN_DATASET, type=AssetTypes.MLTABLE, path=str(train_dir),
                 description="AESO RAI train split (model features + target).")
        )
        test_asset = ml_client.data.create_or_update(
            Data(name=RAI_TEST_DATASET, type=AssetTypes.MLTABLE, path=str(test_dir),
                 description="AESO RAI test split (model features + target).")
        )
        print(f"  registered {train_asset.name}:{train_asset.version} / "
              f"{test_asset.name}:{test_asset.version}")

        mlflow_dir = tmp_path / "mlflow-model"
        signature = infer_signature(train_df[feature_cols], model.predict(train_df[feature_cols]))
        mlflow.sklearn.save_model(
            model, path=str(mlflow_dir), signature=signature,
            input_example=train_df[feature_cols].head(5),
        )
        model_asset = ml_client.models.create_or_update(
            Model(name=RAI_MODEL_NAME, path=str(mlflow_dir), type=AssetTypes.MLFLOW_MODEL,
                  description="MLflow-flavored AESO forecaster used for the Responsible AI dashboard.")
        )
        print(f"  registered MLflow model {model_asset.name}:{model_asset.version}")

    return {
        "model_name": model_asset.name, "model_version": model_asset.version,
        "train_name": train_asset.name, "train_version": train_asset.version,
        "test_name": test_asset.name, "test_version": test_asset.version,
    }


def resolve_assets(ml_client) -> dict:
    """Look up the latest versions of previously prepared RAI assets."""
    model = ml_client.models.get(name=RAI_MODEL_NAME, label="latest")
    train = ml_client.data.get(name=RAI_TRAIN_DATASET, label="latest")
    test = ml_client.data.get(name=RAI_TEST_DATASET, label="latest")
    return {
        "model_name": model.name, "model_version": model.version,
        "train_name": train.name, "train_version": train.version,
        "test_name": test.name, "test_version": test.version,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-test-rows", type=int, default=5000,
                        help="Cap rows RAI explains (keeps the dashboard job fast).")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--skip-prep", action="store_true",
                        help="Reuse already-registered RAI model + datasets.")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    from azure.ai.ml import Input, MLClient, dsl
    from azure.identity import DefaultAzureCredential

    cfg = load_resources()
    ml_client = get_ml_client()

    if args.skip_prep:
        assets = resolve_assets(ml_client)
    else:
        assets = prepare_assets(ml_client, args.test_fraction, Path(args.csv))

    # Pull the official Responsible AI components from the public registry.
    registry = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        registry_name=RAI_REGISTRY,
    )
    rai_constructor = registry.components.get(
        name="rai_tabular_insight_constructor", label="latest")
    rai_explanation = registry.components.get(
        name="rai_tabular_explanation", label="latest")
    rai_erroranalysis = registry.components.get(
        name="rai_tabular_erroranalysis", label="latest")
    rai_gather = registry.components.get(
        name="rai_tabular_insight_gather", label="latest")

    model_info = f"{assets['model_name']}:{assets['model_version']}"
    compute = cfg["compute_cluster"]

    @dsl.pipeline(
        name="aeso_responsible_ai_dashboard",
        description="Responsible AI dashboard (explanations + error analysis) for the AESO price forecaster.",
        compute=compute,
    )
    def rai_pipeline():
        construct = rai_constructor(
            title="AESO Price Forecaster - Responsible AI",
            task_type="regression",
            model_info=model_info,
            model_input=Input(type="mlflow_model",
                              path=f"azureml:{assets['model_name']}:{assets['model_version']}"),
            train_dataset=Input(type="mltable",
                                path=f"azureml:{assets['train_name']}:{assets['train_version']}"),
            test_dataset=Input(type="mltable",
                               path=f"azureml:{assets['test_name']}:{assets['test_version']}"),
            target_column_name=TARGET_COLUMN,
            categorical_column_names=json.dumps(["is_weekend", "is_holiday"]),
            maximum_rows_for_test_dataset=args.max_test_rows,
        )
        construct.set_limits(timeout=900)

        explain = rai_explanation(
            comment="Global & per-row feature explanations (SHAP).",
            rai_insights_dashboard=construct.outputs.rai_insights_dashboard,
        )
        erroranalysis = rai_erroranalysis(
            rai_insights_dashboard=construct.outputs.rai_insights_dashboard,
        )
        # The gather step uploads the dashboard and links it to the model via
        # `model_info`; we intentionally don't surface its `path`-typed outputs
        # as pipeline-level outputs (not a valid top-level asset type).
        rai_gather(
            constructor=construct.outputs.rai_insights_dashboard,
            insight_1=explain.outputs.explanation,
            insight_2=erroranalysis.outputs.error_analysis,
        )

    pipeline_job = rai_pipeline()
    submitted = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="aeso-responsible-ai"
    )
    print(f"Submitted RAI dashboard job: {submitted.name}")
    print(f"Studio URL: {submitted.studio_url}")
    print(f"Dashboard will attach to model '{model_info}' (Studio > Models > "
          f"{assets['model_name']} > Responsible AI).")

    if args.no_stream:
        return
    ml_client.jobs.stream(submitted.name)
    completed = ml_client.jobs.get(submitted.name)
    print(f"RAI pipeline finished with status: {completed.status}")


if __name__ == "__main__":
    main()
