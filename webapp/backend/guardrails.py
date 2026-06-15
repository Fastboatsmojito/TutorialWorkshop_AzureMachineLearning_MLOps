"""Guardrails for the ETRM agent: Prompt Shields + Content Safety + domain.

Three layers, applied around every agent turn (all additive — if Content Safety
isn't configured or a call fails, the agent still answers):

  1. Prompt Shields (input)   detects jailbreak / prompt-injection attacks before
                              the user's text ever reaches the LLM.
  2. Content moderation (out) screens the model's reply for hate / violence /
                              sexual / self-harm above a severity threshold.
  3. Domain guardrail (input) keeps the assistant on-topic (AESO power-price
                              forecasting) and out of financial/trading advice.

Auth: this tenant disables account keys, so we call Content Safety with an AAD
token (DefaultAzureCredential) — the same identity the agent uses for the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import requests

from config import settings

# Content Safety severities are 0,2,4,6. Block the model's OUTPUT at medium+ (>=4).
OUTPUT_SEVERITY_BLOCK = 4
CONTENT_SAFETY_API_VERSION = "2024-09-01"

SAFE_INPUT_REPLY = (
    "I can't help with that request. I'm the ETRM Forecast Assistant — I answer "
    "questions about the Alberta (AESO) day-ahead power price using our deployed "
    "model. Try asking for tomorrow's forecast, a what-if scenario, or model accuracy."
)
SAFE_OUTPUT_REPLY = (
    "I'm not able to share that response. Let's keep things focused on AESO "
    "power-price forecasting — ask me for a forecast, a scenario, or model metrics."
)

# Domain guardrail: block clearly off-topic asks. Conservative on purpose so it
# never rejects a legitimate power-price question.
_OFFTOPIC_PATTERNS = [
    r"\b(stock|stocks|equit(y|ies)|crypto|bitcoin|ethereum)\b",
    r"\b(medical|diagnos(e|is)|symptom|prescription|legal advice|lawsuit)\b",
    r"\b(write|compose|draft)\b.{0,20}\b(poem|story|song|essay|novel|joke)\b",
    r"\b(recipe|cook|bake)\b",
]
# These energy/forecasting terms make a prompt clearly in-domain (allowlist wins).
_INDOMAIN_PATTERNS = [
    r"\b(price|forecast|aeso|power|electricity|demand|load|mwh?|mw|grid)\b",
    r"\b(temperature|wind|gas|scenario|peak|hedge|model|accuracy|metric)\b",
]


@dataclass
class GuardrailResult:
    allowed: bool = True
    triggered: list[dict[str, Any]] = field(default_factory=list)

    def trip(self, name: str, detail: dict[str, Any]) -> None:
        self.allowed = False
        self.triggered.append({"guardrail": name, **detail})


def _token() -> str | None:
    try:
        from azure.identity import DefaultAzureCredential

        cred = DefaultAzureCredential()
        return cred.get_token("https://cognitiveservices.azure.com/.default").token
    except Exception:
        return None


def _cs_url(path: str) -> str | None:
    base = (settings.content_safety_endpoint or "").rstrip("/")
    if not base:
        return None
    return f"{base}/contentsafety/{path}?api-version={CONTENT_SAFETY_API_VERSION}"


def prompt_shield(user_text: str) -> dict[str, Any]:
    """Call Prompt Shields. Returns {'attack_detected': bool, 'available': bool}."""
    url = _cs_url("text:shieldPrompt")
    token = _token()
    if not url or not token:
        return {"attack_detected": False, "available": False}
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"userPrompt": user_text, "documents": []},
            timeout=10,
        )
        resp.raise_for_status()
        analysis = resp.json().get("userPromptAnalysis", {})
        return {"attack_detected": bool(analysis.get("attackDetected")), "available": True}
    except Exception:
        return {"attack_detected": False, "available": False}


def moderate_text(text: str) -> dict[str, Any]:
    """Call Content Safety text moderation. Returns max severity per category."""
    url = _cs_url("text:analyze")
    token = _token()
    if not url or not token:
        return {"available": False, "max_severity": 0, "categories": {}}
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"text": text[:10000], "outputType": "FourSeverityLevels"},
            timeout=10,
        )
        resp.raise_for_status()
        cats = {c["category"]: c.get("severity", 0) for c in resp.json().get("categoriesAnalysis", [])}
        return {"available": True, "max_severity": max(cats.values(), default=0), "categories": cats}
    except Exception:
        return {"available": False, "max_severity": 0, "categories": {}}


def domain_check(user_text: str) -> dict[str, Any]:
    """Lightweight on-topic check. In-domain signals override off-topic ones."""
    text = user_text.lower()
    in_domain = any(re.search(p, text) for p in _INDOMAIN_PATTERNS)
    off_topic = any(re.search(p, text) for p in _OFFTOPIC_PATTERNS)
    blocked = off_topic and not in_domain
    return {"off_topic": blocked, "in_domain": in_domain}


def check_input(user_text: str) -> GuardrailResult:
    """Run input guardrails (Prompt Shields + domain) before calling the model."""
    result = GuardrailResult()
    if not settings.guardrails_enabled:
        return result

    shield = prompt_shield(user_text)
    result.triggered.append({"guardrail": "prompt_shield", "detail": shield, "blocked": shield["attack_detected"]})
    if shield["attack_detected"]:
        result.allowed = False

    domain = domain_check(user_text)
    result.triggered.append({"guardrail": "domain", "detail": domain, "blocked": domain["off_topic"]})
    if domain["off_topic"]:
        result.allowed = False

    return result


def check_output(reply_text: str) -> GuardrailResult:
    """Run output moderation on the model's reply before returning it."""
    result = GuardrailResult()
    if not settings.guardrails_enabled or not reply_text:
        return result

    mod = moderate_text(reply_text)
    blocked = mod["max_severity"] >= OUTPUT_SEVERITY_BLOCK
    result.triggered.append({"guardrail": "content_safety", "detail": mod, "blocked": blocked})
    if blocked:
        result.allowed = False
    return result
