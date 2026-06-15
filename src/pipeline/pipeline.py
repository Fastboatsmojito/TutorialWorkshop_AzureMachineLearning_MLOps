"""Define and submit the AESO forecasting MLOps pipeline.

prep -> train -> evaluate (quality gate). On success, registers the model from
the pipeline's output (gate-then-register: models are promoted on merit).

This is the artifact you show live for the "MLOps" pillar of the demo:
a reproducible, parameterized, cached, auditable training pipeline.

Usage:
    python src/pipeline/pipeline.py            # submit + stream + register on pass
    python src/pipeline/pipeline.py --no-register
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add src/
from common.workspace import get_ml_client, load_resources  # noqa: E402

SRC_DIR = Path(__file__).resolve().parents[1]  # the `src` folder
TRAIN_DIR = SRC_DIR / "training"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmse-threshold", type=float, default=80.0)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    from azure.ai.ml import Input, Output, command, dsl
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Environment, Model

    cfg = load_resources()
    ml_client = get_ml_client()

    env = ml_client.environments.create_or_update(
        Environment(
            name="aeso-train-env",
            conda_file=str(TRAIN_DIR / "conda_train.yml"),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        )
    )
    compute = cfg["compute_cluster"]

    prep = command(
        name="prep_data",
        display_name="Prep & validate AESO data",
        code=str(SRC_DIR),
        command="python pipeline/prep.py --input_data ${{inputs.raw}} --output_data ${{outputs.clean}}",
        environment=env,
        inputs={"raw": Input(type=AssetTypes.URI_FILE)},
        outputs={"clean": Output(type=AssetTypes.URI_FOLDER)},
    )
    train = command(
        name="train_model",
        display_name="Train LightGBM forecaster",
        code=str(SRC_DIR),
        command=(
            "python training/train.py --data ${{inputs.clean}} "
            "--model_output ${{outputs.model}}"
        ),
        environment=env,
        inputs={"clean": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"model": Output(type=AssetTypes.URI_FOLDER)},
    )
    evaluate = command(
        name="evaluate_gate",
        display_name="Evaluation quality gate",
        code=str(SRC_DIR),
        command=(
            "python pipeline/evaluate.py --model_dir ${{inputs.model}} "
            "--rmse_threshold ${{inputs.threshold}} --output_dir ${{outputs.gate}}"
        ),
        environment=env,
        inputs={"model": Input(type=AssetTypes.URI_FOLDER), "threshold": Input(type="number")},
        outputs={"gate": Output(type=AssetTypes.URI_FOLDER)},
    )

    @dsl.pipeline(
        name="aeso_price_forecast_pipeline",
        description="ETRM day-ahead AESO price forecast: prep -> train -> evaluate gate",
        default_compute=compute,
    )
    def forecast_pipeline(raw_data, rmse_threshold):
        p = prep(raw=raw_data)
        t = train(clean=p.outputs.clean)
        e = evaluate(model=t.outputs.model, threshold=rmse_threshold)
        return {"trained_model": t.outputs.model, "gate": e.outputs.gate}

    # Resolve the data asset to an explicit version. The `@latest` label
    # shorthand goes through a data-container endpoint that 404s on this
    # workspace's API version, so we look up the latest version directly.
    da_name = cfg["data_asset_name"]
    latest_version = max(
        (int(d.latest_version) for d in ml_client.data.list() if d.name == da_name),
        default=None,
    )
    if latest_version is None:
        raise SystemExit(
            f"Data asset '{da_name}' not found in workspace {cfg['workspace']}. "
            f"Run `python data/ingest_data.py` first."
        )
    data_asset = ml_client.data.get(name=da_name, version=str(latest_version))

    pipeline_job = forecast_pipeline(
        raw_data=Input(
            type=AssetTypes.URI_FILE,
            path=data_asset.id,
        ),
        rmse_threshold=args.rmse_threshold,
    )

    submitted = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="aeso-price-forecast"
    )
    print(f"Submitted pipeline job: {submitted.name}")
    print(f"Studio URL: {submitted.studio_url}")

    if args.no_stream:
        return

    ml_client.jobs.stream(submitted.name)
    completed = ml_client.jobs.get(submitted.name)
    print(f"Pipeline finished with status: {completed.status}")

    if args.no_register or completed.status != "Completed":
        return

    print("Registering model from pipeline output...")
    model = ml_client.models.create_or_update(
        Model(
            name=cfg["model_name"],
            path=f"azureml://jobs/{submitted.name}/outputs/trained_model",
            type=AssetTypes.CUSTOM_MODEL,
            description="AESO price forecaster registered via MLOps pipeline (gate passed).",
            tags={"pipeline_job": submitted.name},
        )
    )
    print(f"Registered {model.name}:{model.version}")


if __name__ == "__main__":
    main()
