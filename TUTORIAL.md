# Tutorial — Azure ML + MLOps + Foundry + Evals & Guardrails

This repository is a complete, runnable example of taking a real forecasting
problem all the way to a governed, production AI service with a natural-language
interface. The tutorial turns that example into a guided learning path.

We forecast the **Alberta (AESO) day-ahead hourly power price**, productionize it
with **Azure Machine Learning** and **MLOps**, put a **Foundry GPT-4o agent** in
front of it, and then make that agent trustworthy with **Foundry evaluations** and
**guardrails**.

## Choose your journey

There are two ways through this material. Pick the one that fits your time.

| Journey | File | Best for |
|---|---|---|
| Self-paced tutorial | this file + [tutorial/](tutorial/) | Learning each layer hands-on, at your own pace. No time limit. |
| 90-minute demo | [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | Presenting the end-to-end story to an audience. Pre-run the slow jobs, narrate live. |

Both journeys use the **same code and the same Azure resources**. The tutorial
explains the *why* and lets you run each piece; the demo script is a presenter
run-of-show.

## What you will learn

The tutorial is organized into four pillars, each building on the last.

1. Azure Machine Learning: workspaces, versioned data assets, training with
   experiment tracking, and a managed online endpoint.
2. MLOps: a reproducible pipeline with an evaluation gate, a model registry with
   lineage, and Azure DevOps CI/CD.
3. Foundry: a GPT-4o agent that calls the ML model as a tool, so every number it
   reports comes from the model and not the language model.
4. Trust: Foundry evaluations (quality, safety, and a custom tool-accuracy
   evaluator) and runtime guardrails (Prompt Shields, Content Safety, and a
   domain guardrail).

## Module map

Work through these in order. Each module has a `README.md` (the narrative) and,
where there is something to run, a notebook you execute on Azure ML compute.

| Module | Topic | Notebook |
|---|---|---|
| [00 Overview](tutorial/00-overview.md) | Architecture, prerequisites, cost, and teardown | — |
| [01 Azure ML foundations](tutorial/01-azure-ml-foundations/README.md) | Workspace, `MLClient`, versioned data assets | yes |
| [02 Training and tracking](tutorial/02-training-and-tracking/README.md) | Features, time-series split, LightGBM, MLflow | yes |
| [03 MLOps pipeline and gate](tutorial/03-mlops-pipeline-gate/README.md) | `prep → train → evaluate gate → register` | yes |
| [04 Deploy the endpoint](tutorial/04-deploy-endpoint/README.md) | Managed online endpoint, scenario scoring | yes |
| [05 Foundry agent](tutorial/05-foundry-agent/README.md) | GPT-4o function calling over the model | yes |
| [06 Foundry evaluations](tutorial/06-foundry-evals/README.md) | Quality, safety, custom tool-accuracy | yes |
| [07 Guardrails](tutorial/07-guardrails/README.md) | Prompt Shields, Content Safety, domain | yes |
| [08 Evals in CI/CD](tutorial/08-evals-in-cicd/README.md) | The Foundry eval gate in Azure DevOps | — |
| [99 Responsible AI](tutorial/99-responsible-ai/README.md) | RAI dashboard for the forecaster | yes |

## Before you start

Read [tutorial/00-overview.md](tutorial/00-overview.md) first. It covers the
prerequisites, provisions the Azure resources, and explains how to run the
notebooks on Azure ML compute. Everything in this tutorial runs against live
Azure, so you need a subscription and the Azure CLI.

When you finish, run `infra/teardown.ps1` to stop billing.
