"""Quick smoke test for the deployed AML online endpoint.

Reads the scoring URI + key from .azure-resources.json and prints a sample
forecast. Use this in pre-flight before the demo.

    python tools/test_endpoint.py
    python tools/test_endpoint.py --horizon 168 --demand-multiplier 1.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

RES = Path(__file__).resolve().parents[1] / ".azure-resources.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--demand-multiplier", type=float, default=1.0)
    parser.add_argument("--temperature-offset", type=float, default=0.0)
    args = parser.parse_args()

    cfg = json.loads(RES.read_text())
    url = cfg.get("aml_endpoint_url")
    key = cfg.get("aml_endpoint_key")
    if not url or not key:
        raise SystemExit("Endpoint URL/key not found in .azure-resources.json. Deploy first.")

    payload = {"horizon_hours": args.horizon}
    if args.start:
        payload["start"] = args.start
    if args.demand_multiplier != 1.0:
        payload["demand_multiplier"] = args.demand_multiplier
    if args.temperature_offset:
        payload["temperature_offset_c"] = args.temperature_offset

    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print("Units:", data["units"], "| Model:", data["model"])
    print("Summary:", json.dumps(data["summary"], indent=2))
    print("First 3 hours:")
    for row in data["forecast"][:3]:
        print("  ", row["timestamp"], f"${row['predicted_price']}/MWh")
    print(f"... ({len(data['forecast'])} hours total)")


if __name__ == "__main__":
    main()
