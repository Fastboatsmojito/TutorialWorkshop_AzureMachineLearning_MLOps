# Module 04 — Deploy the managed online endpoint

This module serves the model behind a REST endpoint so anything, including the
Foundry agent, can get a forecast. You also meet the scenario knobs that make the
endpoint useful for what-if analysis.

Notebook: [04_endpoint_and_scenarios.ipynb](04_endpoint_and_scenarios.ipynb)

## Concepts

### Managed online endpoints

[src/deploy/deploy_endpoint.py](../../src/deploy/deploy_endpoint.py) registers the
model and creates a managed online endpoint named `etrm-forecast` with a single
`blue` deployment taking 100 percent of traffic. Azure manages the compute,
scaling, and TLS. After a successful deploy, the script writes the scoring URI and
key into `.azure-resources.json` so the web app and the evaluators can find it.

### The scoring contract

[src/training/score.py](../../src/training/score.py) defines the request and
response. A request asks for a horizon and, optionally, a start time and scenario
adjustments. The response is an hourly forecast in CAD/MWh plus a summary with the
average, the peak hour, and the scenario that produced it.

### Climatology fills the gaps

A trader rarely supplies every driver for every future hour. When drivers are
missing, the endpoint fills them from a climatology table: the average demand,
temperature, wind, and gas by month and hour. You always get a sensible forecast.

### Scenario knobs

Two parameters power what-if analysis:

* `demand_multiplier` scales grid demand, for example 1.10 for plus 10 percent.
* `temperature_offset_c` shifts temperature, for example minus 8 for a cold snap.

These are the same knobs the chatbot uses later when a trader asks for a cold-snap
scenario in plain English.

## What you will do

1. Deploy (or confirm) the `etrm-forecast` endpoint.
2. Call it for a 24-hour forecast and read the summary.
3. Run a cold-snap scenario and compare the prices.

## Key takeaways

* A managed online endpoint turns a model artifact into a service.
* The scoring contract and climatology make the endpoint robust to partial input.
* Scenario knobs let one model answer many what-if questions.

## Next

Continue to [Module 05 — The Foundry agent](../05-foundry-agent/README.md).
