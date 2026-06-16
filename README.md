# ETRM Forecasting on Azure ML + Foundry LLM — TC Energy Demo

A complete, runnable, **Azure-only** demo that shows the full path from raw data to a
production ML service with a natural-language interface:

1. **Azure Machine Learning** forecasts a real ETRM quantity — the
   **Alberta (AESO) day-ahead hourly power price**.
2. The **MLOps** capabilities of AML — versioned data assets, MLflow experiment
   tracking, a reproducible training **pipeline with an evaluation gate**, the
   model registry, a managed online **endpoint**, AutoML, and **Azure DevOps** CI/CD.
3. A **Foundry LLM agent** (gpt-4o) layered on top of the deployed model: a simple
   web app where you ask natural-language questions and the LLM calls the ML model
   via **function calling** so every number comes from the model, not the LLM.
4. **Trust** — **Foundry evaluations** (quality, safety, and a custom tool-accuracy
   evaluator) that run on AML compute and report into the Foundry portal, plus
   runtime **guardrails** (Prompt Shields, Content Safety moderation, and a domain
   guardrail) on the agent.

Everything is provisioned in resource group **`rg-tc-aml-foundry-etrm`** (region
`eastus2`).

> **Live demo web app:** `https://etrm-qa.icybush-1b89aba9.eastus2.azurecontainerapps.io/`

---

## Understand the use case

Before the code, here is the real-world problem this demo solves and the data
behind it.

### What is ETRM trading

ETRM stands for Energy Trading and Risk Management. Energy companies buy and sell
commodities such as power, natural gas, and crude oil, and they manage the
financial risk that comes with volatile prices. An ETRM desk owns the full trade
lifecycle: capturing trades, tracking positions, scheduling physical delivery,
valuing the portfolio, and measuring risk such as Value at Risk (VaR). Accurate
price forecasts sit at the center of that work because they drive hedging
decisions, position limits, and how the desk prices the risk it carries.

### What is AESO

AESO is the Alberta Electric System Operator. It runs Alberta's wholesale
electricity market and operates the provincial power grid. Alberta uses an
energy-only market with a single hourly "pool price" that reflects provincial
supply and demand in real time. That price can swing sharply with weather,
demand, and generation availability, so forecasting it well is valuable to anyone
trading or hedging Alberta power.

### What data we use and what we forecast

We use an AESO-style hourly dataset, bundled at
[data/aeso_hourly.csv](data/aeso_hourly.csv) and reproducible with
[data/generate_dataset.py](data/generate_dataset.py). Each row is one hour and
captures the price along with the drivers that move it.

| Column | What it is |
|---|---|
| `pool_price` | Alberta pool price in CAD/MWh. This is the value we forecast. |
| `ail_demand_mw` | Alberta Internal Load: total provincial electricity demand |
| `temperature_c` | Temperature, the main driver of heating and cooling demand |
| `wind_generation_mw` | Wind output, which lowers price through the merit-order effect |
| `gas_price_aeco` | AECO natural gas price, the fuel cost of the marginal generator |

From these we derive calendar features (hour, day of week, season, holidays) and
**net load** (demand minus wind), the single biggest price driver.

The model forecasts the **AESO day-ahead hourly pool price in CAD/MWh** for the
next 24 hours. A trader can ask "what will tomorrow's prices look like, and when
do they peak?" then use the answer to plan hedges and positions.

---

## How to use this repo

Pick the path that fits your goal. The first two use the same code and Azure
resources in this repository.

| Resource | Best for |
|---|---|
| [Machine Learning crash course (video)](https://youtu.be/1vkb7BCMQd0?si=YqTuzF1baRbcGwIH) | Brand new to ML. Watch this first: a 6.5-hour lecture that builds the core machine learning concepts from the ground up, so the rest of this repo makes sense. |
| [TUTORIAL.md](TUTORIAL.md) | Learning hands-on at your own pace. A self-paced walkthrough of each layer (Azure ML, MLOps, the Foundry agent, evals, and guardrails) with a notebook per topic. |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | Presenting live. The 90-minute presenter run-of-show for delivering the end-to-end story to an audience. |
| [AML_MLOps_101](https://github.com/Fastboatsmojito/AML_MLOps_101) | Starting from the basics. A companion beginner workshop that teaches Azure ML and the MLOps lifecycle (data versioning, experiment tracking, registration, deployment, monitoring, and pipelines) on simpler text and tabular use cases. Begin here if you want the 101 before this end-to-end demo. |

Just want to run it? Jump to [Quick start (from scratch)](#quick-start-from-scratch).

---

## Architecture

```
 AESO-style hourly data ─► ADLS (workspace store) ─► AML Data Asset (versioned)
                                                        │
        ┌───────────────── AML Pipeline ───────────────┴──────────┐
        │  prep ─► train (LightGBM) ─► evaluate gate ─► register   │
        └──────────────────────────────────────────────┬──────────┘
                          MLflow tracking + Model Registry
                                                        │
                          AML Managed Online Endpoint (blue)  ◄── REST scoring
                                                        ▲
                                                        │ REST (function tool call)
                          Azure AI Foundry agent (gpt-4o, Entra ID auth)
                                                        │
                          FastAPI + static Q&A web app  ──►  Azure Container Apps
                          (user-assigned managed identity for ACR pull + Foundry)
```

The custom **LightGBM** model backs the live endpoint. An **AutoML Forecasting**
job (`src/automl/`) runs alongside to showcase AML's low-code model sweeping and
leaderboard.

---

## Deployed Azure resources

| Resource | Name | Purpose |
|---|---|---|
| Resource group | `rg-tc-aml-foundry-etrm` | Holds everything (eastus2) |
| AML workspace | `mlw-tc-etrm` | Training, registry, endpoints, MLflow |
| AML compute cluster | `cpu-cluster` | Pipeline / AutoML compute (min 0 nodes) |
| Data asset | `aeso-hourly-prices` | Versioned training dataset |
| Registered model | `aeso-price-forecaster` | LightGBM forecaster |
| Online endpoint | `etrm-forecast` | Managed online endpoint (blue) |
| Foundry (AIServices) | `foundry-tc-etrm` | Hosts the `gpt-4o` deployment |
| Model deployment | `gpt-4o` | LLM used by the agent |
| Eval judge deployment | `gpt-4o-eval` | Separate LLM judge for evaluations (isolates demo traffic) |
| Foundry project | `proj-tc-etrm` | Where evaluation runs appear in the Foundry portal |
| Container Registry | `acrtcetrm1c92281a` | Stores the web app image |
| Container Apps env | `cae-tc-etrm` | Hosts the web app |
| Container App | `etrm-qa` | The Q&A web app (external ingress) |
| Managed identity | `id-etrm-webapp` | ACR pull + Foundry access (no keys) |

Live values (endpoint URLs, keys, identity client id) are written to
`.azure-resources.json` at the repo root by the provisioning/deploy scripts. **That
file is git-ignored because it contains the endpoint key.**

---

## Repo layout

| Path | What it is |
|---|---|
| `data/generate_dataset.py` | Reproducible AESO-style hourly dataset generator |
| `data/ingest_data.py` | Registers the CSV as a versioned AML data asset |
| `data/aeso_hourly.csv` | The bundled demo dataset (committed) |
| `src/common/features.py` | Shared feature engineering (no train/serve skew) |
| `src/common/workspace.py` | `MLClient` helper (reads `.azure-resources.json`) |
| `src/training/train.py` | LightGBM training + MLflow logging + artifacts |
| `src/training/score.py` | Online endpoint scoring script (+ scenario knobs) |
| `src/training/conda_*.yml` | Conda envs for training / scoring |
| `src/automl/automl_job.py` | AML AutoML forecasting job |
| `src/pipeline/` | `prep` → `train` → `evaluate` gate → register pipeline |
| `src/deploy/deploy_endpoint.py` | Register model + deploy managed online endpoint |
| `src/evals/` | Foundry evaluations: quality, safety, custom tool-accuracy, gate runner |
| `webapp/backend/` | FastAPI app: `agent.py` (Foundry + tools), `guardrails.py`, `aml_client.py`, `config.py`, `app.py` |
| `webapp/frontend/` | Static chat UI (HTML/CSS/JS + Chart.js via CDN) |
| `webapp/Dockerfile` | Container image for the Q&A app (built by ACR) |
| `mlops/` | Azure DevOps CI/CD YAML (CI, CD, and the Foundry eval gate) + setup guide |
| `infra/provision.ps1` | Provision AML + Foundry account/project + gpt-4o + gpt-4o-eval + Content Safety + compute |
| `infra/deploy_containerapp.ps1` | ACR cloud build + deploy web app to Container Apps |
| `infra/teardown.ps1` | Delete resources to stop billing |
| `tools/test_endpoint.py` | Smoke-test the live online endpoint |
| `tests/test_pipeline.py` | Unit + smoke tests (features, train, score) |
| `tests/test_evals.py` | Unit tests for the custom tool-accuracy evaluator |
| `TUTORIAL.md` | Self-paced tutorial entry point (choose your journey) |
| `tutorial/` | Hands-on modules (README + notebook per topic) |
| `DEMO_SCRIPT.md` | The 90-minute presenter run-of-show |

---

## Quick start (from scratch)

> Shell is **PowerShell**. You must be logged in: `az login` (with the `az ml` and
> `containerapp` CLI extensions installed).

```powershell
# 0) Python env
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1) Provision Azure (idempotent): AML workspace, Foundry + gpt-4o, compute
./infra/provision.ps1

# 2) Data: generate the dataset and register it as a versioned AML data asset
python data/generate_dataset.py
python data/ingest_data.py

# 3) Train locally (fast) — produces model_artifacts/ (model.pkl, metrics, model_card)
python src/training/train.py --data data/aeso_hourly.csv --model_output model_artifacts

# 4) Deploy the model to a managed online endpoint (~6-10 min)
python src/deploy/deploy_endpoint.py --model-dir model_artifacts

# 5) Deploy the Q&A web app to Azure Container Apps
#    (ACR cloud build -> Container App; managed identity for ACR pull + Foundry)
./infra/deploy_containerapp.ps1
```

`deploy_containerapp.ps1` prints the public URL, e.g.
`https://etrm-qa.<id>.eastus2.azurecontainerapps.io/`.

### MLOps demo (the "show me the pipeline" part)

```powershell
python src/pipeline/pipeline.py --rmse-threshold 80   # prep -> train -> gate -> register
python src/automl/automl_job.py                        # (optional) AutoML leaderboard
```

### Smoke-test the live endpoint

```powershell
python tools/test_endpoint.py
```

### Run the web app locally

```powershell
cd webapp/backend
..\..\.venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
# open http://localhost:8000  (uses your az login for Foundry auth)
```

---

## How the LLM agent works

The agent (`webapp/backend/agent.py`) is **gpt-4o with function calling**. The LLM
handles language and reasoning; the ML model is the source of truth for numbers.
The system prompt forbids the model from inventing prices — it must call a tool.

| Tool | What it does |
|---|---|
| `get_forecast` | Calls the AML online endpoint for the AESO day-ahead hourly forecast. Supports scenario analysis via `demand_multiplier` and `temperature_offset_c`. |
| `get_model_metrics` | Returns the deployed model's MAE / RMSE / sMAPE / R² and metadata from the model card. |
| `explain_price_drivers` | Returns the model's top feature importances plus the driver values (net load, temperature, gas price) for the peak hour. |

The backend returns the assistant reply, a forecast **chart** payload (rendered with
Chart.js), and a **tool trace** so the audience can see exactly which tool was called.

Example questions for the demo:
- "What's the day-ahead power price forecast for tomorrow, and when does it peak?"
- "Show me a cold-snap scenario: temperature down 8°C and demand up 10%."
- "How accurate is this model?"
- "What's driving the price on the peak hour?"

---

## Evaluations & guardrails

The agent is measured before it ships and protected while it runs.

**Foundry evaluations** (`src/evals/`) run on AML compute and log to the Foundry
project so results appear in the portal's Evaluations tab:

- Quality (LLM-judge on the **separate** `gpt-4o-eval` deployment): groundedness,
  relevance, coherence.
- Safety (Azure RAI service): violence, hate/unfairness, self-harm, sexual, and
  indirect-attack (jailbreak) susceptibility.
- A **custom tool-accuracy evaluator** (`tool_accuracy_eval.py`): proves the agent
  called the ML model with the right args, and avoided a tool call on off-topic
  prompts. This is the "no hallucinated prices" property, measured.

`run_evals.py` enforces a pass/fail gate (groundedness, relevance, coherence,
tool-accuracy, zero safety defects) and `mlops/azure-pipelines-evals.yml` runs it
as an Azure DevOps gate. Run it with:

```powershell
python src/evals/run_evals.py --mode remote   # submit to cpu-cluster, log to Foundry
```

**Guardrails** (`webapp/backend/guardrails.py`) wrap every agent turn:

- Prompt Shields screen the user's input for jailbreak / prompt injection.
- Content Safety moderation screens the model's reply for harmful content.
- A domain guardrail keeps the assistant on AESO forecasting, not financial advice.

Guardrails are additive (keyless, via Entra ID): if Content Safety is unavailable,
the agent still answers and the local domain check still fires.

See the [tutorial](TUTORIAL.md) modules 06–08 for the full walkthrough.

---

## CI/CD (Azure DevOps)

`mlops/` contains pipeline YAML and a setup guide:

- `azure-pipelines-ci.yml` — lint + unit tests on PRs.
- `azure-pipelines-cd.yml` — on merge to `main`: retrain via the AML pipeline,
  enforce the evaluation gate, register the model, and (re)deploy the endpoint.
- `azure-pipelines-evals.yml` — on merge to `main` (agent / eval changes): run the
  Foundry eval gate on AML compute and fail if quality/safety thresholds aren't met.

See `mlops/README.md` for service-connection and variable-group setup.

---

## Authentication notes

This tenant disables local/key auth on storage and on the Foundry account, so the
demo uses **Entra ID + managed identity** end-to-end (best practice):

- AML workspace datastores use **identity-based** access (no storage keys).
- The web app calls the Foundry model with **AAD tokens** — a **user-assigned
  managed identity** (`id-etrm-webapp`) on the Container App; locally it uses your
  `az login`. **No API keys for the LLM.**
- The same managed identity pulls the container image from ACR (`AcrPull`), so no
  registry passwords are stored.
- The AML online endpoint uses key auth, passed to the web app as a Container App
  environment variable (kept out of git via `.azure-resources.json`).

---

## Cost & cleanup

Small SKUs throughout (compute scales to 0 nodes, Container App scales to 1 small
replica, Basic ACR, 1× `Standard_DS2_v2` endpoint instance). To stop billing:

```powershell
./infra/teardown.ps1            # deletes the whole resource group (fastest)
./infra/teardown.ps1 -KeepGroup # or just the expensive pieces (endpoint, compute, container app, ACR)
```

---

## Troubleshooting

- **`az acr build` crashes on Windows** with a `cp1252`/`charmap` `UnicodeEncodeError`.
  This is a console log-streaming bug — the cloud build still succeeds. The deploy
  script suppresses the stream and polls the ACR run result instead.
- **Container App stuck "Activating"** right after deploy: it's pulling the image on
  first start; give it ~30–60s. Check logs with
  `az containerapp logs show -n etrm-qa -g rg-tc-aml-foundry-etrm --tail 40 --type console`.
- **`mlflow` dependency conflict** on install: `mlflow` is pinned to `2.15.1` to
  satisfy `azureml-mlflow`.
- **AutoML job fails with "Identity of the specified managed compute ... is not
  found"**: the compute cluster needs a managed identity (AutoML reads the dataset
  as the cluster's identity). `provision.ps1` now creates `cpu-cluster` with a
  system-assigned identity and grants it `Storage Blob Data Contributor`. To fix an
  existing cluster: `az ml compute update -n cpu-cluster -w mlw-tc-etrm -g rg-tc-aml-foundry-etrm --identity-type SystemAssigned`.
- **AutoML submit fails with `AuthorizationFailure` uploading the dataset**: an
  Azure Policy may set the workspace storage account to *public network access =
  Disabled*, which blocks the SDK's upload from your laptop. Temporarily re-enable
  it: `az storage account update --ids <storageId> --public-network-access Enabled`,
  submit, then it can be set back to Disabled.
