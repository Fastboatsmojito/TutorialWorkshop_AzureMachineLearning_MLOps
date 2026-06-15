"""Pipeline step 1: data preparation / validation.

Reads the raw AESO data asset, performs basic cleaning and sanity checks,
and writes a cleaned CSV to the step output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_csv(p: Path) -> Path:
    if p.is_dir():
        csvs = list(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV under {p}")
        return csvs[0]
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_data", required=True)
    args = parser.parse_args()

    src = resolve_csv(Path(args.input_data))
    df = pd.read_csv(src, parse_dates=["timestamp"]).sort_values("timestamp")

    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df[(df["pool_price"] >= 0) & (df["pool_price"] <= 999.99)]
    # Forward-fill any small gaps in exogenous drivers.
    df[["ail_demand_mw", "temperature_c", "wind_generation_mw", "gas_price_aeco"]] = (
        df[["ail_demand_mw", "temperature_c", "wind_generation_mw", "gas_price_aeco"]].ffill().bfill()
    )
    df = df.dropna(subset=["pool_price"])
    after = len(df)

    out_dir = Path(args.output_data)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "aeso_clean.csv", index=False)
    print(f"Prep complete: {before:,} -> {after:,} rows after cleaning")


if __name__ == "__main__":
    main()
