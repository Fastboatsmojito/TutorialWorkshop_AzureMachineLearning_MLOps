"""Generate a realistic AESO-style hourly power market dataset.

Why synthetic? For a *reliable* live demo we cannot depend on an external API
being reachable on the day. This generator produces a high-fidelity, fully
reproducible (seeded) hourly series for the Alberta market with the same
structure a real AESO extract would have:

    timestamp, pool_price, ail_demand_mw, temperature_c,
    wind_generation_mw, gas_price_aeco

The relationships are deliberately realistic:
  * Alberta Internal Load (AIL): winter/summer seasonality + daily double peak
    + lower weekends.
  * Temperature: annual sinusoid + daily swing + noise (Alberta range).
  * Wind: autocorrelated, suppresses price (merit order effect).
  * AECO gas price: slow random walk (marginal unit fuel cost).
  * Pool price: convex function of NET load (demand - wind) scaled by gas,
    with realistic evening spikes, floored at 0 and capped at the AESO
    offer cap of $999.99/MWh.

`data/ingest_data.py` can instead pull the *real* AESO series if you have
access; this keeps the demo self-contained by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OFFER_CAP = 999.99
SEED = 42


def _annual(ts: pd.DatetimeIndex, peak_doy: int) -> np.ndarray:
    """Sinusoid peaking at a given day-of-year (1.0 at peak, -1.0 opposite)."""
    doy = ts.dayofyear.to_numpy()
    return np.cos(2 * np.pi * (doy - peak_doy) / 365.25)


def generate(start: str, end: str, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    n = len(ts)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    is_weekend = dow >= 5

    # ---- Temperature (Celsius) ----
    # Cold winters (peak cold ~ Jan 15), warm summers; daily swing; noise.
    temp_season = -16 * _annual(ts, peak_doy=15)  # +16 summer, -16 winter offset
    temp_daily = 5 * np.sin(2 * np.pi * (hour - 9) / 24)  # warmest mid-afternoon
    temperature = 6 + temp_season + temp_daily + rng.normal(0, 3, n)

    # ---- Alberta Internal Load (MW) ----
    base_load = 9700
    # Heating (cold) and cooling (hot) both raise load -> U-shape in temp.
    thermal_load = 55 * np.maximum(0, 18 - temperature) + 35 * np.maximum(0, temperature - 22)
    # Daily double-peak (morning + evening), weekends lower.
    daily_shape = (
        260 * np.sin(2 * np.pi * (hour - 8) / 24)
        + 180 * np.sin(2 * np.pi * (hour - 19) / 24)
    )
    weekend_drop = np.where(is_weekend, -650, 0)
    ail_demand = base_load + thermal_load + daily_shape + weekend_drop + rng.normal(0, 180, n)
    ail_demand = np.clip(ail_demand, 7000, 12500)

    # ---- Wind generation (MW): autocorrelated AR(1) process ----
    wind = np.zeros(n)
    w = 600.0
    for i in range(n):
        w = 0.92 * w + 0.08 * rng.uniform(0, 1800) + rng.normal(0, 60)
        wind[i] = min(max(w, 0), 2200)

    # ---- AECO gas price ($/GJ): slow random walk ----
    gas = np.zeros(n)
    g = 2.6
    for i in range(n):
        g += rng.normal(0, 0.015)
        g = min(max(g, 1.2), 5.5)
        gas[i] = g

    # ---- Pool price ($/MWh) ----
    net_load = ail_demand - wind
    # Normalised scarcity 0..1 over a plausible net-load band.
    scarcity = np.clip((net_load - 7000) / (12000 - 7000), 0, 1)
    # Convex supply curve: price rises steeply as scarcity -> 1.
    base_price = 18 + 9 * gas  # marginal fuel cost component
    price = base_price + 240 * scarcity**3 + 35 * scarcity
    # Evening peak premium and weekday premium.
    price += 22 * np.maximum(0, np.sin(2 * np.pi * (hour - 18) / 24))
    price += np.where(is_weekend, -8, 4)
    price += rng.normal(0, 6, n)

    # Occasional scarcity spikes (tight supply events ~0.6% of hours), correlated
    # with scarcity so they remain partly learnable rather than pure noise.
    spike_mask = rng.uniform(0, 1, n) < 0.006
    price[spike_mask] += rng.uniform(80, 350, spike_mask.sum()) * (0.5 + scarcity[spike_mask])

    price = np.clip(price, 0, OFFER_CAP)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "pool_price": np.round(price, 2),
            "ail_demand_mw": np.round(ail_demand, 1),
            "temperature_c": np.round(temperature, 1),
            "wind_generation_mw": np.round(wind, 1),
            "gas_price_aeco": np.round(gas, 3),
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AESO-style hourly dataset")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "aeso_hourly.csv"),
    )
    args = parser.parse_args()

    df = generate(args.start, args.end, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(df.describe(include="all").T[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
