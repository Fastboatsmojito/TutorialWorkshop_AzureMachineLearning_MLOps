"""Pipeline step 3: evaluation gate.

Reads metrics.json produced by training and enforces a quality gate. Emits a
`gate.json` describing whether the model is eligible to be registered/deployed.
This is the heart of the MLOps story: models are promoted on merit, not by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--rmse_threshold", type=float, default=80.0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    metrics_path = Path(args.model_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    rmse = metrics["rmse"]
    passed = rmse <= args.rmse_threshold

    gate = {
        "passed": bool(passed),
        "rmse": rmse,
        "rmse_threshold": args.rmse_threshold,
        "mae": metrics.get("mae"),
        "smape": metrics.get("smape"),
        "r2": metrics.get("r2"),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2))

    try:
        mlflow.log_metric("gate_passed", int(passed))
        mlflow.log_metric("gate_rmse", rmse)
    except Exception:
        pass

    print(json.dumps(gate, indent=2))
    if not passed:
        raise SystemExit(
            f"Evaluation gate FAILED: rmse {rmse:.2f} > threshold {args.rmse_threshold:.2f}"
        )
    print("Evaluation gate PASSED")


if __name__ == "__main__":
    main()
