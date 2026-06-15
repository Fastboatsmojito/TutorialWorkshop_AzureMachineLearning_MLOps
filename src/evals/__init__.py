"""Foundry evaluations for the ETRM forecasting agent.

This package adds the *evaluation* pillar on top of the deployed Foundry agent:

  - quality_evals.py      groundedness / relevance / coherence (LLM-judge)
  - safety_evals.py       violence / hate / self-harm / jailbreak (Azure RAI service)
  - tool_accuracy_eval.py custom evaluator: did the agent call the ML model
                          (and with the right args) instead of inventing numbers?
  - dataset.py            the curated ETRM question set the suite runs against
  - agent_runner.py       runs the live agent to produce responses + tool traces
  - run_evals.py          orchestrates the run, logs to the Foundry project,
                          and enforces a quality/safety gate (CI-friendly)

Everything is designed to run on AML compute (see run_evals.py --mode remote).
"""
