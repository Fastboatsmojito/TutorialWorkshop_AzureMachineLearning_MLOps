"""Unit tests for the custom tool-accuracy evaluator (no Azure required)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evals"))


def test_correct_tool_and_args():
    from tool_accuracy_eval import ToolCallAccuracyEvaluator

    ev = ToolCallAccuracyEvaluator()
    out = ev(
        query="forecast tomorrow",
        tool_trace=[{"tool": "get_forecast", "args": {"horizon_hours": 24}}],
        expected_tool="get_forecast",
        expected_tool_args={"horizon_hours": 24},
    )
    assert out["tool_selection"] == 1.0
    assert out["tool_args_match"] == 1.0
    assert out["tool_accuracy"] == 1.0


def test_wrong_args_partial_credit():
    from tool_accuracy_eval import ToolCallAccuracyEvaluator

    ev = ToolCallAccuracyEvaluator()
    out = ev(
        query="cold snap",
        tool_trace=[{"tool": "get_forecast", "args": {"horizon_hours": 24}}],
        expected_tool="get_forecast",
        expected_tool_args={"horizon_hours": 24, "demand_multiplier": 1.10},
    )
    assert out["tool_selection"] == 1.0
    assert out["tool_args_match"] == 0.0  # demand_multiplier missing


def test_offtopic_should_not_call_tool():
    from tool_accuracy_eval import ToolCallAccuracyEvaluator

    ev = ToolCallAccuracyEvaluator()
    good = ev(query="buy stock?", tool_trace=[], expected_tool="")
    assert good["tool_accuracy"] == 1.0

    bad = ev(
        query="buy stock?",
        tool_trace=[{"tool": "get_forecast", "args": {}}],
        expected_tool="",
    )
    assert bad["tool_accuracy"] == 0.0


def test_tool_trace_accepts_json_string():
    from tool_accuracy_eval import ToolCallAccuracyEvaluator

    ev = ToolCallAccuracyEvaluator()
    out = ev(
        query="metrics?",
        tool_trace='[{"tool": "get_model_metrics", "args": {}}]',
        expected_tool="get_model_metrics",
        expected_tool_args={},
    )
    assert out["tool_selection"] == 1.0
