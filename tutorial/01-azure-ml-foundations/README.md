# Module 01 — Azure ML foundations

This module connects you to the Azure ML workspace and registers the training
data as a versioned asset. These two ideas, a workspace client and versioned
data, are the foundation everything else builds on.

Notebook: [01_workspace_and_data.ipynb](01_workspace_and_data.ipynb)

## Concepts

### The workspace

An Azure ML workspace is the top-level container for your ML work: data assets,
compute, jobs, models, and endpoints all live inside it. You interact with it
through `MLClient` from the `azure-ai-ml` SDK.

The helper in [src/common/workspace.py](../../src/common/workspace.py) builds an
`MLClient` for you. It reads `.azure-resources.json` (written by `provision.ps1`),
falls back to sensible defaults, and lets environment variables override both, so
the same code runs locally and in CI.

### Versioned data assets

A data asset is a named, versioned pointer to data in the workspace. Versioning
gives you lineage: every training run records the exact data version it consumed,
so you can always answer "which data produced this model?".

[data/ingest_data.py](../../data/ingest_data.py) uploads the bundled
`aeso_hourly.csv` to the workspace datastore and registers it as the `uri_file`
asset `aeso-hourly-prices`. The dataset is hourly from 2024 to 2026 with the
columns a real AESO extract has: `pool_price`, `ail_demand_mw`, `temperature_c`,
`wind_generation_mw`, and `gas_price_aeco`.

The data is identity-based: the datastore uses your Entra ID, not a storage key,
which matches the security posture of a real trading environment.

## What you will do

1. Build an `MLClient` and confirm it can reach the workspace.
2. Register (or refresh) the `aeso-hourly-prices` data asset.
3. List the asset's versions and inspect its tags to see lineage in action.

## Key takeaways

* `MLClient` is your handle to everything in the workspace.
* Registering data as a versioned asset is what makes later training runs
  auditable and reproducible.
* Identity-based access means no secrets in the data path.

## Next

Continue to [Module 02 — Training and tracking](../02-training-and-tracking/README.md).
