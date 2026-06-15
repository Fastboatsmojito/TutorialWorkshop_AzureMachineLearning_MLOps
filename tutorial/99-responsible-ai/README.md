# Module 99 — Responsible AI dashboard

This optional module builds an Azure ML Responsible AI dashboard for the
forecaster: global and per-row explanations plus error analysis, attached to the
registered model in Studio. It rounds out the trust story on the ML side, the way
evaluations and guardrails do on the agent side.

Notebook: [99_rai_dashboard.ipynb](99_rai_dashboard.ipynb)

Code: [src/pipeline/rai_dashboard.py](../../src/pipeline/rai_dashboard.py)

## Concepts

### What the dashboard shows

The Responsible AI dashboard answers two questions about the model:

* Explanations: which features drive predictions, globally and for individual
  rows, using SHAP. You can confirm that net load, temperature, and gas price are
  the real drivers, which builds trust with the desk.
* Error analysis: where the model is least accurate, so you know which conditions
  (for example scarcity spikes) carry the most risk.

### How it is built

The Responsible AI components need an MLflow-flavored model and MLTable train and
test splits whose columns are exactly the model's feature matrix plus the target.
[src/pipeline/rai_dashboard.py](../../src/pipeline/rai_dashboard.py) prepares both,
registers them, and then wires the official Responsible AI components from the
public `azureml` registry into a pipeline:

```text
constructor ─► explanation + error analysis ─► gather
```

The job runs on the `cpu-cluster`. When it finishes, the dashboard attaches to the
`aeso-price-forecaster-rai` model under Studio, Models, Responsible AI.

## What you will do

1. Submit the RAI pipeline to the `cpu-cluster`.
2. Open the dashboard in Studio and read the global explanations.
3. Use error analysis to find the model's weakest regions.

## Key takeaways

* Explanations turn a black box into something a trader can question.
* Error analysis tells you where the model's risk concentrates.
* Responsible AI on the model complements evaluations and guardrails on the agent.

## Next

Return to the [tutorial index](../../TUTORIAL.md) or present the
[90-minute demo](../../DEMO_SCRIPT.md).
