# Module 08 — Evaluations in CI/CD

The final step is to make evaluation automatic. This module wires the Foundry
eval gate into Azure DevOps so the agent is held to the same promote-on-merit
standard as the ML model.

There is no notebook here: CI/CD is configuration and process. You read the
pipeline, set up the variables, and let it run on merges.

Pipeline: [mlops/azure-pipelines-evals.yml](../../mlops/azure-pipelines-evals.yml)

## Concepts

### Two gates, one philosophy

The repository now has two gates, one per layer:

* The model gate in [src/pipeline/evaluate.py](../../src/pipeline/evaluate.py)
  blocks a model that fails its RMSE threshold (Module 03).
* The agent gate in [src/evals/run_evals.py](../../src/evals/run_evals.py) blocks
  an agent that fails quality, safety, or tool-accuracy thresholds (Module 06).

Both express the same idea: nothing reaches production unless it earns its way
there, and the decision is recorded.

### The pipeline

[mlops/azure-pipelines-evals.yml](../../mlops/azure-pipelines-evals.yml) triggers
on merges to `main` that touch the agent or the eval suite. It installs the
authoring dependencies, then runs the eval suite on AML compute through an
`AzureCLI@2` task:

```bash
python src/evals/run_evals.py --mode remote
```

In remote mode the script submits an Azure ML command job to the `cpu-cluster`.
The gate is enforced inside that job, so a gate failure fails the job, which fails
the pipeline step. The agent is not promoted.

### Configuration

The pipeline reads its configuration from the `aml-etrm-vars` variable group.
Beyond the workspace coordinates, the eval gate needs the Foundry project
endpoint, the app and judge deployment names, the Content Safety endpoint, and the
ML endpoint URL and key (the key marked secret). The full list and the required
role assignments are in [mlops/README.md](../../mlops/README.md).

Why Azure DevOps and not GitHub Actions? This project standardizes on Azure
DevOps for CI/CD, alongside the existing CI and CD pipelines in the same folder.

## What you will do

1. Read the eval-gate pipeline and trace how a gate failure stops promotion.
2. Add the required variables to the `aml-etrm-vars` group.
3. Create the pipeline in Azure DevOps and run it once manually.

## Key takeaways

* The agent gets the same automated quality gate as the model.
* Remote mode keeps evaluation on AML compute, even from CI.
* Two gates, model and agent, give end-to-end promotion control.

## Next

Continue to [Module 99 — Responsible AI](../99-responsible-ai/README.md), or
return to the [tutorial index](../../TUTORIAL.md).
