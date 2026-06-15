"""The Foundry LLM agent: gpt-4o with function-calling tools over the AML model.

The model never invents prices. When asked anything quantitative it must call
`get_forecast` (which hits the deployed AML endpoint) or `get_model_metrics`.
This is the "LLM layer over the ML model" pattern: the LLM handles language and
reasoning; the ML model is the source of truth for numbers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from openai import AzureOpenAI

import aml_client
import guardrails
from config import settings

SYSTEM_PROMPT = """You are the ETRM Forecast Assistant for a North American energy trading desk.
You answer questions about the Alberta (AESO) day-ahead hourly power price using a
deployed machine-learning model.

Rules:
- For ANY numeric question about prices or forecasts, you MUST call get_forecast.
  Never invent or estimate prices yourself.
- For questions about model accuracy/quality, call get_model_metrics.
- For "what is driving price" questions, call explain_price_drivers.
- Prices are in CAD/MWh. Be concise and decision-oriented, like talking to a trader.
- When you present a forecast, mention the average, the peak hour, and the key driver
  (e.g. high net load / cold temperature / low wind), and note the model name.
- Today's date for reference is {today}.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get the AESO day-ahead hourly power price forecast from the deployed ML model. Supports scenario analysis via demand_multiplier and temperature_offset_c.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "ISO start datetime, e.g. 2026-01-15T00:00:00. Optional; defaults to the next hour.",
                    },
                    "horizon_hours": {
                        "type": "integer",
                        "description": "Number of hours to forecast (1-720). Use 24 for a day, 168 for a week.",
                    },
                    "demand_multiplier": {
                        "type": "number",
                        "description": "Scale grid demand for scenarios, e.g. 1.10 for +10% load. Default 1.0.",
                    },
                    "temperature_offset_c": {
                        "type": "number",
                        "description": "Shift temperature in Celsius for scenarios, e.g. -5 for a cold snap. Default 0.",
                    },
                },
                "required": ["horizon_hours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": "Return the deployed model's evaluation metrics (MAE, RMSE, sMAPE, R2) and metadata.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_price_drivers",
            "description": "Explain what drives the price forecast: returns the model's top feature importances plus the driver values for a given day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "ISO start datetime for the day to explain. Optional.",
                    }
                },
            },
        },
    },
]


def _build_client() -> AzureOpenAI:
    if settings.aoai_api_key:
        return AzureOpenAI(
            azure_endpoint=settings.aoai_endpoint,
            api_key=settings.aoai_api_key,
            api_version=settings.aoai_api_version,
        )
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=settings.aoai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=settings.aoai_api_version,
    )


def _execute_tool(name: str, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if name == "get_forecast":
        result = aml_client.get_forecast(**args)
        state["last_forecast"] = result  # for charting
        return result
    if name == "get_model_metrics":
        card = settings.model_card()
        return {"model": card.get("name"), "metrics": card.get("metrics", {}),
                "top_features": card.get("top_features", [])}
    if name == "explain_price_drivers":
        importances = settings.feature_importances()
        top = dict(list(importances.items())[:8])
        drivers = {}
        try:
            fc = aml_client.get_forecast(start=args.get("start"), horizon_hours=24)
            state["last_forecast"] = fc
            peak = max(fc["forecast"], key=lambda r: r["predicted_price"])
            drivers = {
                "peak_hour": peak["timestamp"],
                "peak_price": peak["predicted_price"],
                "net_load_proxy_mw": round(peak["ail_demand_mw"] - peak["wind_generation_mw"], 1),
                "temperature_c": peak["temperature_c"],
                "gas_price_aeco": peak["gas_price_aeco"],
            }
        except Exception:
            pass
        return {"top_feature_importances": top, "day_drivers": drivers}
    return {"error": f"unknown tool {name}"}


def run_agent(messages: list[dict[str, str]], max_iters: int = 5,
              return_tool_outputs: bool = False) -> dict[str, Any]:
    """Run a function-calling conversation turn. Returns reply + chart + tool trace.

    Set return_tool_outputs=True (used by the evaluation harness) to also get the
    raw tool results, so groundedness can be scored against what the model returned.
    """
    state: dict[str, Any] = {}
    tool_trace: list[dict[str, Any]] = []
    tool_outputs: list[dict[str, Any]] = []

    # --- Input guardrails: Prompt Shields (jailbreak) + domain (on-topic) ------
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    gin = guardrails.check_input(last_user)
    guardrail_trace: list[dict[str, Any]] = list(gin.triggered)
    if not gin.allowed:
        out = {"reply": guardrails.SAFE_INPUT_REPLY, "chart": None,
               "tool_trace": tool_trace, "guardrails": guardrail_trace}
        if return_tool_outputs:
            out["tool_outputs"] = tool_outputs
        return out

    def _finalize(reply: str) -> dict[str, Any]:
        # Output guardrail: moderate the model's reply before returning it.
        gout = guardrails.check_output(reply)
        trace = guardrail_trace + list(gout.triggered)
        final_reply = reply if gout.allowed else guardrails.SAFE_OUTPUT_REPLY
        chart = _make_chart(state.get("last_forecast")) if gout.allowed else None
        out = {"reply": final_reply, "chart": chart,
               "tool_trace": tool_trace, "guardrails": trace}
        if return_tool_outputs:
            out["tool_outputs"] = tool_outputs
        return out

    client = _build_client()

    convo: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=datetime.now(timezone.utc).date().isoformat())}
    ]
    convo.extend(messages)

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model=settings.aoai_deployment,
            messages=convo,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return _finalize(msg.content or "")

        convo.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(tc.function.name, args, state)
            tool_trace.append({"tool": tc.function.name, "args": args})
            tool_outputs.append({"tool": tc.function.name, "args": args, "output": result})
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                }
            )

    # Fallback if max_iters exceeded.
    return _finalize("I gathered the data but ran out of reasoning steps. Please ask again.")


def _make_chart(forecast: dict[str, Any] | None) -> dict[str, Any] | None:
    if not forecast or "forecast" not in forecast:
        return None
    rows = forecast["forecast"]
    return {
        "labels": [r["timestamp"] for r in rows],
        "prices": [r["predicted_price"] for r in rows],
        "summary": forecast.get("summary", {}),
        "units": forecast.get("units", "CAD/MWh"),
    }
