# Module 00 — Overview, setup, and teardown

Start here. This module gets your environment ready, provisions the Azure
resources the rest of the tutorial uses, and explains how to run the notebooks on
Azure ML compute.

## The problem we solve

Day-ahead price uncertainty is the core ETRM (Energy Trading and Risk Management)
problem: it drives hedging decisions, position limits, and value-at-risk. We
forecast the Alberta (AESO) day-ahead hourly pool price in CAD/MWh, a real
quantity a trading desk acts on every day.

## Architecture

```text
 AESO-style hourly data ─► ADLS (workspace store) ─► AML Data Asset (versioned)
                                                        │
        ┌───────────────── AML Pipeline ───────────────┴──────────┐
        │  prep ─► train (LightGBM) ─► evaluate GATE ─► register   │
        └──────────────────────────────────────────────┬──────────┘
                          MLflow tracking + Model Registry
                                                        │
                          AML Managed Online Endpoint (blue)  ◄── REST scoring
                                                        ▲
                                                        │ REST (function tool call)
                          Azure AI Foundry agent (gpt-4o, Entra ID auth)
                                                        │
                          Foundry evaluations + guardrails  ◄── quality / safety
                                                        │
                          FastAPI + static Q&A web app  ──►  Azure Container Apps
```

## The four pillars

1. Azure ML provides the workspace, the versioned data, the training compute, the
   model registry, and the endpoint that serves predictions.
2. MLOps wraps training in a reproducible pipeline with a quality gate, then
   automates retrain and redeploy through Azure DevOps.
3. Foundry hosts the GPT-4o model. The agent calls the ML endpoint as a tool, so
   the language model never invents numbers.
4. Trust adds Foundry evaluations (is the agent grounded, safe, and calling the
   model correctly?) and guardrails (Prompt Shields, Content Safety moderation,
   and an on-topic domain check).

## Azure resources

Everything lives in resource group `rg-tc-aml-foundry-etrm` in region `eastus2`.

| Resource | Name | Purpose |
|---|---|---|
| AML workspace | `mlw-tc-etrm` | Training, registry, endpoints, MLflow |
| Compute cluster | `cpu-cluster` | Pipeline, AutoML, and eval jobs (min 0 nodes) |
| Data asset | `aeso-hourly-prices` | Versioned training dataset |
| Registered model | `aeso-price-forecaster` | LightGBM forecaster |
| Online endpoint | `etrm-forecast` | Managed online endpoint (blue) |
| Foundry account | `foundry-tc-etrm` | Hosts the model deployments and the project |
| App model | `gpt-4o` | The agent's language model |
| Judge model | `gpt-4o-eval` | A separate deployment used only by evaluations |
| Foundry project | `proj-tc-etrm` | Where evaluation runs appear in the portal |
| Container App | `etrm-qa` | The Q&A web app |

The judge model is a **separate deployment** on purpose: evaluation traffic never
competes with the live demo's model capacity.

## Authentication model

This tenant disables account keys, so the tutorial uses Entra ID end-to-end.

* AML datastores use identity-based access (no storage keys).
* The web app and the evaluators call Foundry with AAD tokens. Locally that is
  your `az login`; in production it is a managed identity. No API keys for the LLM.
* Content Safety (Prompt Shields and moderation) authenticates the same way.
* The AML online endpoint uses key auth; the key is written to the git-ignored
  `.azure-resources.json` and passed to the web app as an environment variable.

## Prerequisites

* An Azure subscription with permission to create resources and assign roles.
* Azure CLI with the `ml` and `containerapp` extensions, logged in with `az login`.
* Python 3.11.
* Quota for a GPT-4o deployment in `eastus2` (two deployments: app and judge).

## Set up your environment

The shell is PowerShell. From the repository root:

```powershell
# 1) Python environment
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Provision Azure (idempotent): AML workspace, Foundry account, gpt-4o and
#    gpt-4o-eval deployments, the Foundry project, Content Safety, and compute.
./infra/provision.ps1
```

`provision.ps1` writes the resource names, endpoints, and the project endpoint to
`.azure-resources.json` at the repository root. That file is git-ignored because
it holds the endpoint key. Every script and notebook in the tutorial reads its
configuration from it.

## Running the notebooks on Azure ML compute

The decision for this tutorial is to run everything on Azure ML compute.

1. Open Azure ML Studio for `mlw-tc-etrm`.
2. Create a small compute instance (for example `Standard_DS3_v2`) under Compute.
3. Open the `tutorial/` folder there, or clone this repo into the compute
   instance, and select the Python 3.11 kernel.
4. The heavy jobs (the training pipeline, AutoML, evaluations, and the RAI
   dashboard) are submitted from the notebooks to the `cpu-cluster`, so the
   notebook host itself stays light.

You can also run the notebooks locally against live Azure if you prefer; the code
is identical because authentication uses `DefaultAzureCredential` either way.

## Cost and teardown

The compute cluster scales to zero when idle, and the endpoint runs a single
small instance. The GPT-4o deployments bill per token. When you are done:

```powershell
./infra/teardown.ps1
```

This deletes the resources so billing stops. Re-running `provision.ps1` recreates
everything.

## Next

Continue to [Module 01 — Azure ML foundations](01-azure-ml-foundations/README.md).
