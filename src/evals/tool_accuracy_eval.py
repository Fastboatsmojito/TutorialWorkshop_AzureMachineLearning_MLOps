"""Custom evaluator: did the agent use the ML model as the source of truth?

The whole point of the agent pattern is that the LLM must NOT invent prices: any
quantitative answer has to come from a tool call into the deployed AML model. A
generic groundedness score doesn't capture this, so we measure it directly from
the tool trace.

azure-ai-evaluation calls an evaluator as `evaluator(**row)` where the row keys
come from the evaluator's column_mapping. We return a flat dict of numeric scores
(0.0-1.0) plus a short reason; the harness aggregates the numbers automatically.
"""
from __future__ import annotations

import json
from typing import Any


def _as_list(tool_trace: Any) -> list[dict[str, Any]]:
    if tool_trace is None or tool_trace == "":
        return []
    if isinstance(tool_trace, str):
        try:
            tool_trace = json.loads(tool_trace)
        except json.JSONDecodeError:
            return []
    if isinstance(tool_trace, dict):
        return [tool_trace]
    return [t for t in tool_trace if isinstance(t, dict)]


def _args_match(expected: dict[str, Any], actual: dict[str, Any], rel_tol: float = 0.02) -> bool:
    """Every expected arg must appear in the actual call within tolerance."""
    for key, exp in expected.items():
        if key not in actual:
            return False
        act = actual[key]
        if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
            tol = max(abs(exp) * rel_tol, 0.01)
            if abs(float(act) - float(exp)) > tol:
                return False
        elif str(act) != str(exp):
            return False
    return True


class ToolCallAccuracyEvaluator:
    """Scores whether the agent called the expected ML tool with the right args.

    Output keys:
      tool_selection   1.0 if the right tool (or correctly NO tool) was used
      tool_args_match  1.0 if the expected args flowed into the call
      grounded_in_model 1.0 if a numeric answer is backed by a model tool call
      tool_accuracy    overall (mean of the above)
    """

    def __init__(self) -> None:
        self._forecasting_tools = {"get_forecast", "get_model_metrics", "explain_price_drivers"}

    def __call__(
        self,
        *,
        query: str = "",
        tool_trace: Any = None,
        expected_tool: str = "",
        expected_tool_args: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls = _as_list(tool_trace)
        called_tools = [c.get("tool") for c in calls]
        expected_tool = (expected_tool or "").strip()

        if expected_tool == "":
            # Off-topic / adversarial: the agent should NOT call a forecasting tool.
            no_tool = not any(t in self._forecasting_tools for t in called_tools)
            reason = (
                "Correctly avoided a forecasting tool call."
                if no_tool
                else f"Unexpectedly called {called_tools} for an off-topic/adversarial prompt."
            )
            return {
                "tool_selection": 1.0 if no_tool else 0.0,
                "tool_args_match": 1.0,  # n/a -> non-penalizing
                "grounded_in_model": 1.0 if no_tool else 0.0,
                "tool_accuracy": 1.0 if no_tool else 0.0,
                "tool_reason": reason,
            }

        selected = expected_tool in called_tools
        # Find the matching call to inspect args.
        match_call = next((c for c in calls if c.get("tool") == expected_tool), None)
        if isinstance(expected_tool_args, str):
            try:
                expected_tool_args = json.loads(expected_tool_args)
            except json.JSONDecodeError:
                expected_tool_args = {}
        expected_tool_args = expected_tool_args or {}
        actual_args = (match_call or {}).get("args", {}) or {}
        args_ok = _args_match(expected_tool_args, actual_args) if selected else False

        grounded = 1.0 if selected else 0.0
        scores = [1.0 if selected else 0.0, 1.0 if args_ok else 0.0, grounded]
        reason = (
            f"Expected '{expected_tool}'; called {called_tools}; "
            f"args_match={args_ok} (expected {expected_tool_args}, got {actual_args})."
        )
        return {
            "tool_selection": 1.0 if selected else 0.0,
            "tool_args_match": 1.0 if args_ok else 0.0,
            "grounded_in_model": grounded,
            "tool_accuracy": round(sum(scores) / len(scores), 3),
            "tool_reason": reason,
        }
