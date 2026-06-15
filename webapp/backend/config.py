"""Configuration for the Q&A web app.

Reads from environment variables (App Service app settings in production).
For local dev it falls back to the repo's .azure-resources.json so you can run
`uvicorn` without exporting anything.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
# Local dev: repo root is two levels up (webapp/backend/..). In the container the
# code is flattened to /app, so fall back to the module dir if that doesn't exist.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    REPO_ROOT = BACKEND_DIR
RESOURCES_FILE = REPO_ROOT / ".azure-resources.json"


def _from_resources() -> dict:
    if RESOURCES_FILE.exists():
        try:
            return json.loads(RESOURCES_FILE.read_text())
        except Exception:
            return {}
    return {}


_res = _from_resources()


class Settings:
    # Azure OpenAI / Foundry
    aoai_endpoint = os.getenv("AOAI_ENDPOINT", _res.get("aoai_endpoint", ""))
    aoai_deployment = os.getenv("AOAI_DEPLOYMENT", _res.get("aoai_deployment", "gpt-4o"))
    aoai_api_version = os.getenv("AOAI_API_VERSION", _res.get("aoai_api_version", "2024-10-21"))
    # Optional key (used only if the account allows key auth; otherwise AAD)
    aoai_api_key = os.getenv("AOAI_API_KEY", "")

    # Separate judge model used by Foundry evaluations (isolates demo traffic).
    aoai_eval_deployment = os.getenv(
        "AOAI_EVAL_DEPLOYMENT", _res.get("aoai_eval_deployment", "gpt-4o-eval")
    )

    # Foundry project (so evaluation runs show up in the Foundry portal).
    foundry_project_endpoint = os.getenv(
        "FOUNDRY_PROJECT_ENDPOINT", _res.get("foundry_project_endpoint", "")
    )
    foundry_project_name = os.getenv(
        "FOUNDRY_PROJECT_NAME", _res.get("foundry_project_name", "")
    )

    # Azure AI Content Safety (Prompt Shields + text moderation). The AIServices
    # account exposes Content Safety at its cognitiveservices.azure.com endpoint.
    content_safety_endpoint = os.getenv(
        "CONTENT_SAFETY_ENDPOINT", _res.get("content_safety_endpoint", "")
    )
    # Guardrails are additive: the agent still runs if this is off or unconfigured.
    guardrails_enabled = os.getenv(
        "GUARDRAILS_ENABLED", str(_res.get("guardrails_enabled", "true"))
    ).lower() in ("1", "true", "yes")

    # AML forecasting endpoint
    aml_endpoint_url = os.getenv("AML_ENDPOINT_URL", _res.get("aml_endpoint_url", ""))
    aml_endpoint_key = os.getenv("AML_ENDPOINT_KEY", _res.get("aml_endpoint_key", ""))

    @staticmethod
    def model_card() -> dict:
        for p in [BACKEND_DIR / "model_card.json", REPO_ROOT / "model_artifacts" / "model_card.json"]:
            if p.exists():
                return json.loads(p.read_text())
        return {"name": "aeso-price-forecaster", "metrics": {}, "top_features": []}

    @staticmethod
    def feature_importances() -> dict:
        for p in [
            BACKEND_DIR / "feature_importances.json",
            REPO_ROOT / "model_artifacts" / "feature_importances.json",
        ]:
            if p.exists():
                return json.loads(p.read_text())
        return {}


settings = Settings()
