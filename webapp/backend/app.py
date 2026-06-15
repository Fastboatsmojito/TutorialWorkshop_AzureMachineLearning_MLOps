"""FastAPI app: serves the Q&A UI and proxies chat to the Foundry LLM agent."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent
from config import settings

def _find_frontend() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here / "frontend", here.parent / "frontend"]:
        if (cand / "index.html").exists():
            return cand
    return here.parent / "frontend"


FRONTEND_DIR = _find_frontend()

app = FastAPI(title="ETRM Forecast Assistant", version="1.0.0")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "aoai_configured": bool(settings.aoai_endpoint),
        "aml_configured": bool(settings.aml_endpoint_url),
        "guardrails_enabled": bool(settings.guardrails_enabled),
        "content_safety_configured": bool(settings.content_safety_endpoint),
        "model": settings.model_card().get("name"),
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    try:
        result = agent.run_agent([m.model_dump() for m in req.messages])
        return JSONResponse(result)
    except Exception as exc:  # surface a friendly error to the UI
        return JSONResponse(
            {"reply": f"Sorry, something went wrong: {exc}", "chart": None, "tool_trace": []},
            status_code=200,
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Serve static assets (css/js) under /static
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
