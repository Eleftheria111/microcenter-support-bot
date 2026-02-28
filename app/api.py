from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import ask

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    return ask(request.message)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/debug/stock")
async def debug_stock(q: str = "tempered glass"):
    """Debug endpoint: shows raw per-store API response for a search query."""
    import requests as req
    import os, json
    from app.agent import _get_api_token, _fetch_with_primp_or_requests, OPENCART_URL

    token = _get_api_token(force_refresh=True)
    if not token:
        return {"error": "Could not get API token", "token": None}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{OPENCART_URL}/",
    }
    status, text = _fetch_with_primp_or_requests(
        f"{OPENCART_URL}/index.php",
        {"route": "api/stock_locations", "api_token": token, "search": q},
        headers,
    )
    try:
        data = json.loads(text)
    except Exception:
        data = {"raw_text": text[:500]}

    return {
        "token_prefix": token[:8] if token else None,
        "api_status": status,
        "response": data,
    }
