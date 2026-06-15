# MLOps with Azure DevOps

This folder contains the CI/CD that turns the notebook-style ML work into a
production, auditable loop. You can **demo the concept from these files** even
without a live Azure DevOps org; if you do have one, the setup is below.

## What the pipelines do

| Pipeline | Trigger | What it does |
|---|---|---|
| `azure-pipelines-ci.yml` | Pull request to `main` | Lint + unit tests + a tiny smoke train/score. Gatekeeps merges. |
| `azure-pipelines-cd.yml` | Merge to `main` | Refresh data asset → run the AML pipeline (`prep → train → evaluate gate → register`) → deploy the registered model to the managed online endpoint. |
| `azure-pipelines-evals.yml` | Merge to `main` (agent / eval changes) | Run the **Foundry eval gate** on AML compute: quality + safety + tool-accuracy evaluators against the live agent. Fails if thresholds aren't met. |

There are **two gates**, one per layer:

- The **model gate** (`src/pipeline/evaluate.py`): a new model is only
  registered/deployed if it beats an RMSE threshold. Lineage is preserved (data
  version → job → model version).
- The **agent gate** (`src/evals/run_evals.py`): the LLM agent is only promoted
  if it stays grounded in the model (tool-accuracy), answers well (groundedness /
  relevance / coherence), and resists harmful + jailbreak prompts. Results are
  logged to the Foundry project so they show in the portal's Evaluations tab.

## One-time setup (only if wiring a live org)

1. **Create an Azure DevOps project** and push this repo to it (Azure Repos) or
   connect your GitHub repo.

2. **Create an ARM service connection** named `aml-etrm-sc`:
   Project Settings → Service connections → New → Azure Resource Manager →
   scope it to the subscription / resource group `rg-tc-aml-foundry-etrm`.
   Grant its service principal these roles on the resource group:
   - `AzureML Data Scientist`
   - `Storage Blob Data Contributor` (on the workspace storage account)

3. **Create a variable group** named `aml-etrm-vars` (Pipelines → Library):
   - `AZURE_SUBSCRIPTION_ID` = `1c92281a-94c3-41ba-b01f-0b238e3c8c0e`
   - `AZURE_RESOURCE_GROUP`  = `rg-tc-aml-foundry-etrm`
   - `AML_WORKSPACE`         = `mlw-tc-etrm`

   The **eval gate** pipeline also needs (copy these from `.azure-resources.json`
   after provisioning; mark `AML_ENDPOINT_KEY` as secret):
   - `FOUNDRY_PROJECT_ENDPOINT` = `https://foundry-tc-etrm.services.ai.azure.com/api/projects/proj-tc-etrm`
   - `AOAI_ENDPOINT`            = `https://foundry-tc-etrm.openai.azure.com/`
   - `AOAI_EVAL_DEPLOYMENT`     = `gpt-4o-eval`
   - `CONTENT_SAFETY_ENDPOINT`  = `https://foundry-tc-etrm.cognitiveservices.azure.com/`
   - `AML_ENDPOINT_URL`         = the scoring URI of `etrm-forecast`
   - `AML_ENDPOINT_KEY`         = the endpoint key (**secret**)

4. **Create three pipelines** pointing at the YAML files in this folder.

   The eval-gate pipeline's service principal also needs data-plane access to the
   Foundry account: `Cognitive Services OpenAI User`, `Cognitive Services User`,
   and `Azure AI Developer` (the same roles `infra/provision.ps1` grants you and
   the compute cluster).

## Demoing without a live org

Open `azure-pipelines-cd.yml` and walk the stages, then run the *same steps*
locally to show the actual mechanics:

```powershell
python src/pipeline/pipeline.py --rmse-threshold 80   # prep -> train -> gate -> register
python src/deploy/deploy_endpoint.py                  # roll out to the endpoint
```

The AML Studio job graph (Jobs → `aeso-price-forecast`) is the visual you show:
the same prep/train/evaluate DAG the CD pipeline would run.

## Running the Foundry eval gate

The eval gate runs on AML compute and logs to the Foundry project. Locally:

```powershell
# Submit the eval suite as an AML job (runs on cpu-cluster, streams back):
python src/evals/run_evals.py --mode remote

# Or run it on your machine against live Azure (faster to iterate):
python src/evals/run_evals.py --mode local
```

A non-zero exit means the gate failed (a metric fell below threshold). Results
appear in the Foundry portal under your project → **Evaluations**. See
[tutorial module 06](../tutorial/06-foundry-evals/README.md) for the walkthrough.

## Responsible AI dashboard

`src/pipeline/rai_dashboard.py` builds an Azure ML **Responsible AI dashboard**
for the forecaster (global + per-row SHAP explanations and error analysis). It
registers an MLflow-flavored model plus MLTable train/test splits, then wires the
official Responsible AI components from the public `azureml` registry:

```
constructor → explanation + error analysis → gather
```

```powershell
python src/pipeline/rai_dashboard.py            # prep assets, submit, stream
python src/pipeline/rai_dashboard.py --skip-prep  # reuse already-registered assets
```

The dashboard attaches to the registered model `aeso-price-forecaster-rai`
(Studio → Models → that model → **Responsible AI** tab).
