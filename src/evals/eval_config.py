"""Resolve evaluation configuration from env vars and/or .azure-resources.json.

Order of precedence for every value: environment variable (UPPER_SNAKE) wins,
then the repo's .azure-resources.json, then a sensible default. This mirrors how
the rest of the repo resolves config and means the SAME script works:
  - locally  (reads .azure-resources.json written by provision.ps1)
  - on AML   (reads env vars passed to the command job; the JSON isn't uploaded)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_FILE = REPO_ROOT / ".azure-resources.json"


def _resources() -> dict:
    if RESOURCES_FILE.exists():
        try:
            return json.loads(RESOURCES_FILE.read_text())
        except Exception:
            return {}
    return {}


def _get(res: dict, key: str, default: str = "") -> str:
    return os.getenv(key.upper()) or res.get(key) or default


@dataclass
class EvalConfig:
    subscription_id: str
    resource_group: str
    workspace: str
    compute_cluster: str
    aoai_endpoint: str
    aoai_eval_deployment: str
    aoai_api_version: str
    foundry_project_endpoint: str
    foundry_project_name: str
    content_safety_endpoint: str
    aml_endpoint_url: str
    aml_endpoint_key: str

    @property
    def judge_model_config(self) -> dict:
        """AzureOpenAIModelConfiguration for the LLM-judge (quality) evaluators.

        No api_key on purpose: this tenant disables key auth, so the evaluators
        authenticate with the ambient AAD identity (your az login locally, or the
        AML compute cluster's managed identity in a remote job).
        """
        return {
            "azure_endpoint": self.aoai_endpoint,
            "azure_deployment": self.aoai_eval_deployment,
            "api_version": self.aoai_api_version,
        }

    @property
    def azure_ai_project(self) -> str:
        """Foundry project endpoint; passing it to evaluate() lights up the
        Evaluations tab in the Foundry portal."""
        return self.foundry_project_endpoint


def get_eval_config() -> EvalConfig:
    res = _resources()
    return EvalConfig(
        subscription_id=_get(res, "subscription_id"),
        resource_group=_get(res, "resource_group", "rg-tc-aml-foundry-etrm"),
        workspace=_get(res, "workspace", "mlw-tc-etrm"),
        compute_cluster=_get(res, "compute_cluster", "cpu-cluster"),
        aoai_endpoint=_get(res, "aoai_endpoint"),
        aoai_eval_deployment=_get(res, "aoai_eval_deployment", "gpt-4o-eval"),
        aoai_api_version=_get(res, "aoai_api_version", "2024-10-21"),
        foundry_project_endpoint=_get(res, "foundry_project_endpoint"),
        foundry_project_name=_get(res, "foundry_project_name", "proj-tc-etrm"),
        content_safety_endpoint=_get(res, "content_safety_endpoint"),
        aml_endpoint_url=_get(res, "aml_endpoint_url"),
        aml_endpoint_key=_get(res, "aml_endpoint_key"),
    )
