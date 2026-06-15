# Module 07 — Guardrails

Evaluations measure the agent before you ship. Guardrails protect it at runtime,
on every request. This module adds three layers around each agent turn and shows
them firing on benign, off-topic, and adversarial inputs.

Notebook: [07_content_safety_prompt_shields.ipynb](07_content_safety_prompt_shields.ipynb)

Code: [webapp/backend/guardrails.py](../../webapp/backend/guardrails.py)

## Concepts

### Three layers, applied around every turn

[webapp/backend/guardrails.py](../../webapp/backend/guardrails.py) wraps the agent
in [webapp/backend/agent.py](../../webapp/backend/agent.py):

1. Prompt Shields screen the user's input for jailbreak and prompt-injection
   attacks before the text reaches the language model.
2. Content Safety moderation screens the model's reply for hate, violence, sexual,
   and self-harm content above a severity threshold before it is returned.
3. A domain guardrail keeps the assistant on-topic (AESO power-price forecasting)
   and out of financial or trading advice.

When an input guardrail trips, the agent returns a safe redirect instead of
calling the model. When the output guardrail trips, the reply is replaced with a
safe message. Every check is recorded and surfaced in the web app, so you can see
which guardrail fired.

### Additive by design

Guardrails are additive. If Content Safety is not configured or a call fails, the
agent still answers, and the domain check (which is local and deterministic) keeps
working. This keeps the demo resilient while still demonstrating defense in depth.

### Keyless, like everything else

Content Safety is called with an AAD token from `DefaultAzureCredential`, the same
identity the agent uses for the model. The AIServices account exposes Content
Safety at its `cognitiveservices.azure.com` endpoint, so no separate resource is
needed.

### How guardrails and evaluations relate

The indirect-attack evaluator in Module 06 measures jailbreak susceptibility in
aggregate, before you ship. Prompt Shields here block jailbreak attempts in real
time, on every request. You want both: measurement to catch regressions, and
runtime enforcement to stop live attacks.

## What you will do

1. Run a benign forecasting question through the input guardrails and watch it
   pass.
2. Run a jailbreak prompt and watch Prompt Shields block it.
3. Run an off-topic prompt and watch the domain guardrail redirect it.
4. Moderate a model reply with Content Safety.

## Key takeaways

* Guardrails are runtime defense; evaluations are pre-ship measurement.
* Three layers (input shield, output moderation, domain) give defense in depth.
* Additive design keeps the agent available even when a check is unavailable.

## Next

Continue to [Module 08 — Evals in CI/CD](../08-evals-in-cicd/README.md).
