"""Helpers to connect to the Azure ML workspace from any script."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_FILE = REPO_ROOT / ".azure-resources.json"

DEFAULTS = {
    "subscription_id": "1c92281a-94c3-41ba-b01f-0b238e3c8c0e",
    "resource_group": "rg-tc-aml-foundry-etrm",
    "workspace": "mlw-tc-etrm",
    "location": "eastus2",
}


def load_resources() -> dict:
    cfg = dict(DEFAULTS)
    if RESOURCES_FILE.exists():
        cfg.update(json.loads(RESOURCES_FILE.read_text()))
    # Environment overrides win (useful in CI).
    for k in list(cfg.keys()):
        env_key = k.upper()
        if os.getenv(env_key):
            cfg[k] = os.getenv(env_key)
    return cfg


def get_ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    cfg = load_resources()
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace"],
    )
