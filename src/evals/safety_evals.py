"""Built-in safety evaluators backed by the Azure AI RAI (Content Safety) service.

Unlike the quality evaluators, these don't use an LLM judge you configure: they
call the managed Responsible AI evaluation service in your Foundry project, which
returns severity-based scores for each harm category.

  Violence / HateUnfairness / SelfHarm / Sexual   content-harm severity
  IndirectAttack                                   jailbreak / prompt-injection susceptibility

They authenticate with the ambient AAD identity (your az login locally, or the
AML compute cluster's managed identity in a remote job).
"""
from __future__ import annotations

from typing import Any


def build_safety_evaluators(azure_ai_project: str) -> dict[str, dict[str, Any]]:
    """Return {name: {"evaluator": <obj>, "column_mapping": {...}}}.

    azure_ai_project is the Foundry project endpoint; the RAI service is scoped to
    that project (and its region).
    """
    from azure.ai.evaluation import (
        HateUnfairnessEvaluator,
        IndirectAttackEvaluator,
        SelfHarmEvaluator,
        SexualEvaluator,
        ViolenceEvaluator,
    )
    from azure.identity import DefaultAzureCredential

    cred = DefaultAzureCredential()
    qcols = {"query": "${data.query}", "response": "${data.response}"}
    attack_cols = {
        "query": "${data.query}",
        "response": "${data.response}",
        "context": "${data.context}",
    }

    def _harm(evaluator_cls):
        return {
            "evaluator": evaluator_cls(credential=cred, azure_ai_project=azure_ai_project),
            "column_mapping": qcols,
        }

    return {
        "violence": _harm(ViolenceEvaluator),
        "hate_unfairness": _harm(HateUnfairnessEvaluator),
        "self_harm": _harm(SelfHarmEvaluator),
        "sexual": _harm(SexualEvaluator),
        "indirect_attack": {
            "evaluator": IndirectAttackEvaluator(
                credential=cred, azure_ai_project=azure_ai_project
            ),
            "column_mapping": attack_cols,
        },
    }
