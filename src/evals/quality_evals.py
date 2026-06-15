"""Built-in quality evaluators (LLM-as-judge) for the agent's answers.

These use the *separate* gpt-4o-eval judge deployment so evaluation traffic never
competes with the live demo's gpt-4o capacity.

  Groundedness  is the answer supported by the tool output (model forecast)?
  Relevance     does the answer address the trader's question?
  Coherence     is the answer well-structured and readable?

Each returns the evaluator instance plus the column_mapping evaluate() needs to
feed it the right fields from our JSONL rows.
"""
from __future__ import annotations

from typing import Any


def build_quality_evaluators(judge_model_config: dict) -> dict[str, dict[str, Any]]:
    """Return {name: {"evaluator": <obj>, "column_mapping": {...}}}.

    judge_model_config is an AzureOpenAIModelConfiguration-shaped dict (endpoint,
    deployment, api_version) pointing at gpt-4o-eval.
    """
    from azure.ai.evaluation import (
        CoherenceEvaluator,
        GroundednessEvaluator,
        RelevanceEvaluator,
    )

    qcols = {"query": "${data.query}", "response": "${data.response}"}
    grounded_cols = {
        "query": "${data.query}",
        "context": "${data.context}",
        "response": "${data.response}",
    }
    return {
        "groundedness": {
            "evaluator": GroundednessEvaluator(judge_model_config),
            "column_mapping": grounded_cols,
        },
        "relevance": {
            "evaluator": RelevanceEvaluator(judge_model_config),
            "column_mapping": qcols,
        },
        "coherence": {
            "evaluator": CoherenceEvaluator(judge_model_config),
            "column_mapping": qcols,
        },
    }
