# Module 02 — Training and experiment tracking

This module trains the LightGBM forecaster and logs the run to MLflow. The focus
is on two things that separate a trustworthy model from a notebook experiment:
shared feature engineering and an honest time-series evaluation.

Notebook: [02_train_and_mlflow.ipynb](02_train_and_mlflow.ipynb)

## Concepts

### One feature module, no train/serve skew

The single most common silent failure in production ML is train/serve skew: the
features computed at training time differ subtly from those computed at inference
time. The fix is to compute features in exactly one place.

[src/common/features.py](../../src/common/features.py) is that one place. Both
training and the online scoring script import `feature_columns()` and
`build_feature_matrix()` from it. The key engineered driver is `net_load_mw`
(demand minus near-zero-cost wind), the biggest determinant of price, alongside
cyclical encodings of hour, day, and month so the model understands that 23:00 is
close to 00:00.

### An honest split for time series

You never shuffle a time series. [src/training/train.py](../../src/training/train.py)
splits chronologically: it trains on the earlier period and evaluates on the
later one, which mirrors how the model is actually used (predict the future from
the past). It logs MAE, RMSE, sMAPE, and R-squared to MLflow.

RMSE comes out higher than MAE because of scarcity price spikes, exactly the tail
risk a trader cares about most.

### Self-contained artifacts plus an MLflow flavor

Training writes a portable artifact folder (`model.pkl`, `climatology.csv`,
`metrics.json`, `feature_importances.json`, and `model_card.json`) that the
endpoint serves directly. It also logs an MLflow model so the registry has rich
metadata.

## What you will do

1. Train the model from the registered data and watch the metrics print.
2. Inspect the artifact folder and the model card.
3. View the run in MLflow, including parameters and metrics.

## Key takeaways

* Share feature code between training and serving to avoid skew.
* Split time series chronologically, never randomly.
* Track every run so models are comparable and reproducible.

## Next

Continue to [Module 03 — MLOps pipeline and gate](../03-mlops-pipeline-gate/README.md).
