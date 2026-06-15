"""Submit an Azure ML AutoML *forecasting* job for AESO pool price.

This showcases AML's low-code AutoML: it featurizes, sweeps many model families
(Prophet, ARIMA, LightGBM, ElasticNet, ensembles...) and produces a leaderboard.
For the demo this is typically PRE-RUN; you walk the results live.

Usage:
    python src/automl/automl_job.py            # submit (returns immediately)
    python src/automl/automl_job.py --stream   # submit and stream logs
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add src/
from common.workspace import get_ml_client, load_resources  # noqa: E402

DATA_CSV = Path(__file__).resolve().parents[2] / "data" / "aeso_hourly.csv"

MLTABLE_YAML = """type: mltable
paths:
  - file: ./aeso_hourly.csv
transformations:
  - read_delimited:
      delimiter: ','
      encoding: utf8
      header: all_files_same_headers
"""


def build_mltable_dir() -> str:
    tmp = Path(tempfile.mkdtemp(prefix="aeso_mltable_"))
    shutil.copy(DATA_CSV, tmp / "aeso_hourly.csv")
    (tmp / "MLTable").write_text(MLTABLE_YAML)
    return str(tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--timeout-minutes", type=int, default=25)
    parser.add_argument("--max-trials", type=int, default=15)
    args = parser.parse_args()

    from azure.ai.ml import Input, automl
    from azure.ai.ml.constants import AssetTypes

    cfg = load_resources()
    ml_client = get_ml_client()

    training_data = Input(type=AssetTypes.MLTABLE, path=build_mltable_dir())

    forecast_job = automl.forecasting(
        compute=cfg["compute_cluster"],
        experiment_name="aeso-automl-forecast",
        training_data=training_data,
        target_column_name="pool_price",
        primary_metric="normalized_root_mean_squared_error",
        n_cross_validations=3,
        enable_model_explainability=True,
    )
    forecast_job.set_forecast_settings(
        time_column_name="timestamp",
        forecast_horizon=args.horizon,
        frequency="H",
    )
    forecast_job.set_limits(
        timeout_minutes=args.timeout_minutes,
        trial_timeout_minutes=6,
        max_trials=args.max_trials,
        enable_early_termination=True,
    )

    submitted = ml_client.jobs.create_or_update(forecast_job)
    print(f"Submitted AutoML job: {submitted.name}")
    print(f"Studio URL: {submitted.studio_url}")

    if args.stream:
        ml_client.jobs.stream(submitted.name)


if __name__ == "__main__":
    main()
