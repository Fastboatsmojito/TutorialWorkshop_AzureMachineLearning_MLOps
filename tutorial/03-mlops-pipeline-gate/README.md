# Module 03 — The MLOps pipeline and the evaluation gate

This module is the difference between a notebook and a trading-grade system. You
submit a reproducible pipeline that prepares data, trains, and then enforces a
quality gate before the model is allowed into the registry.

Notebook: [03_pipeline_and_registry.ipynb](03_pipeline_and_registry.ipynb)

## Concepts

### The pipeline

[src/pipeline/pipeline.py](../../src/pipeline/pipeline.py) defines a three-step
Azure ML pipeline that runs on the `cpu-cluster`:

```text
prep ─► train ─► evaluate gate
```

* `prep` ([src/pipeline/prep.py](../../src/pipeline/prep.py)) does real data
  hygiene: dedupe, clip prices to the valid range, and fill driver gaps.
* `train` reuses the same training code from Module 02.
* `evaluate` is the gate.

The pipeline is parameterized, cached, and recorded as a job you can open in AML
Studio, so anyone can see exactly what ran and on which data version.

### The evaluation gate

[src/pipeline/evaluate.py](../../src/pipeline/evaluate.py) reads the trained
model's metrics and compares RMSE against a threshold. If the model is good
enough, the step passes and the pipeline registers the model. If not, the step
raises and the pipeline stops. Promotion happens on merit, not by hand.

This is the first of two gates in the repository. The second one, in Module 06,
applies the same idea to the LLM agent.

### Lineage in the registry

Each registered version of `aeso-price-forecaster` is tagged with the pipeline
job that produced it, which traces back to the exact data asset version. From any
model you can reach the code, the data, and the metrics that promoted it.

## What you will do

1. Submit the pipeline to the `cpu-cluster` and open the job graph in Studio.
2. Watch the gate pass (or fail if you set the threshold low).
3. List the registered model versions and read their lineage tags.

## Key takeaways

* A pipeline makes training reproducible, parameterized, and auditable.
* The gate enforces quality automatically.
* Registry lineage answers "what produced this model?" for every version.

## Next

Continue to [Module 04 — Deploy the endpoint](../04-deploy-endpoint/README.md).
