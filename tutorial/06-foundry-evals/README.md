# Module 06 — Foundry evaluations

A working agent is not the same as a trustworthy one. This module measures the
agent with Foundry evaluations across three dimensions: quality, safety, and a
custom evaluator that checks the property the whole design depends on, that every
number comes from the model.

Notebook: [06_evals.ipynb](06_evals.ipynb)

Code: [src/evals/](../../src/evals/)

## Concepts

### The evaluation set

[src/evals/dataset.py](../../src/evals/dataset.py) holds the curated questions the
suite grades against. They cover the real cases a desk asks (day-ahead forecast,
a week ahead, a cold-snap scenario, model accuracy, price drivers) plus off-topic
and adversarial prompts so the guardrails get measured, not just the happy path.

### Generating responses on the live agent

[src/evals/agent_runner.py](../../src/evals/agent_runner.py) sends each question to
the same agent the web app uses and records the reply, the tool trace, and the raw
tool outputs. Those tool outputs become the `context` that groundedness is judged
against, so the evaluation reflects what the model actually returned.

### Three kinds of evaluators

Quality evaluators in
[src/evals/quality_evals.py](../../src/evals/quality_evals.py) use an LLM judge:

* Groundedness: is the answer supported by the model's tool output?
* Relevance: does the answer address the trader's question?
* Coherence: is the answer well-structured and readable?

The judge is the **separate** `gpt-4o-eval` deployment, so evaluation traffic
never competes with the live agent's capacity.

Safety evaluators in [src/evals/safety_evals.py](../../src/evals/safety_evals.py)
call the managed Responsible AI service in your Foundry project for violence,
hate and unfairness, self-harm, sexual content, and indirect attack (jailbreak and
prompt injection) susceptibility.

The custom evaluator in
[src/evals/tool_accuracy_eval.py](../../src/evals/tool_accuracy_eval.py) reads the
tool trace and scores whether the agent called the right tool with the right
arguments, and whether off-topic prompts correctly avoided a forecasting call.
This is the "no hallucinated prices" guarantee, now measured instead of assumed.

### Running on AML compute and logging to the portal

[src/evals/run_evals.py](../../src/evals/run_evals.py) orchestrates the run. In
`--mode remote` it submits itself as an Azure ML command job on the `cpu-cluster`,
which is the path this tutorial uses. It passes the Foundry project endpoint to
the evaluation harness, so the results appear in the Foundry portal under your
project's Evaluations tab, with per-row scores you can drill into.

The same script computes a pass/fail gate against thresholds (groundedness,
relevance, coherence, tool accuracy, and a zero-tolerance safety defect rate). A
failing gate exits non-zero, which is what Module 08 uses in CI/CD.

## What you will do

1. Submit the evaluation suite to AML compute with `run_evals.py --mode remote`.
2. Open the run in the Foundry portal and read the quality and safety scores.
3. Inspect the custom tool-accuracy results and the gate summary.

## Key takeaways

* Quality, safety, and tool-accuracy together describe a trustworthy agent.
* A separate judge model keeps evaluation isolated from production traffic.
* Logging to the Foundry project makes results visible and shareable.
* The gate turns evaluation into a promotion decision.

## Next

Continue to [Module 07 — Guardrails](../07-guardrails/README.md).
