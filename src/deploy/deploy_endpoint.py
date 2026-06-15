"""Register the trained model and deploy it to a managed online endpoint.

Creates (or updates) a managed online endpoint with a single 'blue' deployment
serving the LightGBM forecaster via score.py. After success it writes the
scoring URI + key back into .azure-resources.json so the webapp can pick them up.

Usage:
    python src/deploy/deploy_endpoint.py --model-dir model_artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add src/ for `common`
from common.workspace import get_ml_client, load_resources, RESOURCES_FILE  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
TRAIN_DIR = THIS_DIR.parent / "training"

ENDPOINT_NAME = "etrm-forecast"
DEPLOYMENT_NAME = "blue"
INSTANCE_TYPE = "Standard_DS2_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="model_artifacts")
    parser.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    parser.add_argument("--deployment-name", default=DEPLOYMENT_NAME)
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()

    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import (
        CodeConfiguration,
        Environment,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        Model,
    )

    cfg = load_resources()
    ml_client = get_ml_client()

    model_dir = Path(args.model_dir)
    if not (model_dir / "model.pkl").exists():
        raise FileNotFoundError(f"{model_dir}/model.pkl not found. Run training first.")

    print("Registering model...")
    model = ml_client.models.create_or_update(
        Model(
            name=cfg["model_name"],
            path=str(model_dir),
            type=AssetTypes.CUSTOM_MODEL,
            description="LightGBM day-ahead AESO pool-price forecaster.",
        )
    )
    print(f"  registered {model.name}:{model.version}")
    if args.register_only:
        return

    print("Creating/Updating environment...")
    env = ml_client.environments.create_or_update(
        Environment(
            name="aeso-score-env",
            conda_file=str(TRAIN_DIR / "conda_score.yml"),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        )
    )

    print(f"Ensuring endpoint '{args.endpoint_name}'...")
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="ETRM day-ahead AESO price forecast endpoint",
        auth_mode="key",
        tags={"demo": "tc-energy-etrm", "model": cfg["model_name"]},
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Creating deployment '{args.deployment_name}' (this can take ~6-10 min)...")
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=f"{model.name}:{model.version}",
        environment=env,
        code_configuration=CodeConfiguration(code=str(TRAIN_DIR), scoring_script="score.py"),
        instance_type=args.instance_type,
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {args.deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    keys = ml_client.online_endpoints.get_keys(args.endpoint_name)
    scoring_uri = ml_client.online_endpoints.get(args.endpoint_name).scoring_uri

    cfg_out = json.loads(RESOURCES_FILE.read_text()) if RESOURCES_FILE.exists() else {}
    cfg_out.update(
        {
            "online_endpoint": args.endpoint_name,
            "aml_endpoint_url": scoring_uri,
            "aml_endpoint_key": keys.primary_key,
            "model_version": model.version,
        }
    )
    RESOURCES_FILE.write_text(json.dumps(cfg_out, indent=2))

    print("\nDeployment complete.")
    print(f"  scoring_uri: {scoring_uri}")
    print(f"  wrote scoring URI + key to {RESOURCES_FILE.name}")


if __name__ == "__main__":
    main()
