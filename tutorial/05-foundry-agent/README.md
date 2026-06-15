# Module 05 — The Foundry agent

This module puts a GPT-4o agent in front of the model. The pattern is simple and
strict: the language model handles language, and the ML model is the only source
of numbers. The agent reaches the numbers through function calling.

Notebook: [05_foundry_function_calling.ipynb](05_foundry_function_calling.ipynb)

## Concepts

### Function calling, not free text

[webapp/backend/agent.py](../../webapp/backend/agent.py) runs GPT-4o with three
tools. The system prompt forbids the model from inventing prices: any quantitative
question must go through a tool call.

| Tool | What it does |
|---|---|
| `get_forecast` | Calls the AML endpoint for the day-ahead forecast, with optional scenario knobs. |
| `get_model_metrics` | Returns the deployed model's accuracy and metadata from the model card. |
| `explain_price_drivers` | Returns the top feature importances and the driver values for the peak hour. |

The agent loops: the model decides which tool to call, the backend executes it
against the real endpoint, the result goes back into the conversation, and the
model writes a grounded answer. The backend returns the reply, a chart payload,
and a tool trace so you can see exactly which tools ran.

### Keyless authentication

The agent calls Foundry with an AAD token from `DefaultAzureCredential`. Locally
that is your `az login`; in the deployed web app it is a managed identity. There
is no API key for the language model.

### Why this pattern matters

A language model that guesses a power price is worse than useless in a trading
context: it is confidently wrong. By forcing every number through the model, the
agent inherits the model's accuracy and the model's auditability. The next two
modules measure and enforce exactly that property.

## What you will do

1. Send a forecasting question to the agent and read the grounded reply.
2. Inspect the tool trace to confirm the model called `get_forecast`.
3. Ask a what-if scenario and watch the same knobs from Module 04 flow through.

## Key takeaways

* Function calling keeps the language model out of the numbers business.
* The tool trace makes the agent's reasoning auditable.
* Keyless auth keeps secrets out of the agent.

## Next

Continue to [Module 06 — Foundry evaluations](../06-foundry-evals/README.md).
