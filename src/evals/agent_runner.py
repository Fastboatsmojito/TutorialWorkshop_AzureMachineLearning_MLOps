"""Run the live Foundry agent over the eval dataset to produce gradable rows.

For each case we send the question to the SAME agent the web app uses, then record
the final reply, the tool trace, and the raw tool outputs (the model's forecast)
as `context` so groundedness can be judged against what the model actually returned.

The agent lives in webapp/backend, so we add that folder to sys.path and import it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "webapp" / "backend"


def _load_agent():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    import agent  # noqa: E402  (webapp/backend/agent.py)

    return agent


def _context_from_outputs(tool_outputs: list[dict[str, Any]]) -> str:
    if not tool_outputs:
        return "No ML model tool was called for this prompt (non-forecasting or refused)."
    return json.dumps(tool_outputs, default=str)


def generate_rows(cases: list[dict[str, Any]], verbose: bool = True) -> list[dict[str, Any]]:
    """Execute the agent for each case and return JSONL-ready rows."""
    agent = _load_agent()
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"  [{i}/{len(cases)}] {case['id']}: {case['query'][:70]}...")
        try:
            result = agent.run_agent(
                [{"role": "user", "content": case["query"]}], return_tool_outputs=True
            )
            reply = result.get("reply", "")
            tool_trace = result.get("tool_trace", [])
            tool_outputs = result.get("tool_outputs", [])
        except Exception as exc:  # keep the suite running; record the failure
            reply = f"[agent error] {exc}"
            tool_trace, tool_outputs = [], []

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "response": reply,
                "context": _context_from_outputs(tool_outputs),
                "ground_truth": case.get("ground_truth", ""),
                "tool_trace": tool_trace,
                "expected_tool": case.get("expected_tool", ""),
                "expected_tool_args": case.get("expected_tool_args", {}),
                "should_refuse": case.get("should_refuse", False),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return path
