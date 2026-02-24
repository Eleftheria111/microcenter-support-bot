"""
Microcenter.gr AI Support Agent
Uses native OpenAI function calling for Python 3.14 compatibility.
"""
import re
import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

client = OpenAI()

OPENCART_URL = os.getenv("OPENCART_URL", "https://www.microcenter.gr").rstrip("/")
OPENCART_API_KEY = os.getenv("OPENCART_API_KEY", "")
OPENCART_API_USERNAME = os.getenv("OPENCART_API_USERNAME", "default")

STORE_INFO = {
    "Αμπελόκηποι": {"phone": "210 64 68 315"},
    "Παγκράτι":    {"phone": "210 220 1684 ή 211 111 5982"},
}


# ---------------------------------------------------------------------------
# Vectorstore (lazy, loaded once)
# ---------------------------------------------------------------------------

_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = FAISS.load_local(
            "data/faiss", embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore


# ---------------------------------------------------------------------------
# OpenCart API session (token cached for the process lifetime)
# ---------------------------------------------------------------------------

_api_token = None


def _get_api_token() -> str:
    global _api_token
    if _api_token:
        return _api_token
    try:
        resp = requests.post(
            f"{OPENCART_URL}/index.php?route=api/login",
            data={"username": OPENCART_API_USERNAME, "key": OPENCART_API_KEY},
            timeout=10,
        )
        print(f"[OpenCart auth] status={resp.status_code} body={resp.text[:300]}")
        data = resp.json()
        _api_token = data.get("api_token") or data.get("token", "")
    except Exception as e:
        _api_token = ""
        print(f"[OpenCart auth] failed: {e}")
    return _api_token


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str) -> str:
    docs = _get_vectorstore().similarity_search(query, k=6)
    if not docs:
        return "Δεν βρέθηκαν αποτελέσματα στη βάση γνώσεων."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title', '')}]\nURL: {d.metadata.get('url', '')}\n{d.page_content}"
        for d in docs
    )


def _store_stock_line(store_name: str, qty: int) -> str:
    """Format a single store's stock status line per business rules."""
    info = STORE_INFO.get(store_name, {})
    phone = info.get("phone", "")
    if qty > 2:
        return f"✅ Υπάρχει στους {store_name}"
    if qty in (1, 2):
        phone_str = f" — καλέστε για κράτηση: {phone}" if phone else ""
        return f"⚠️ Περιορισμένο απόθεμα στο {store_name}{phone_str}"
    return f"❌ Δεν υπάρχει στο {store_name}"


def _fetch_with_primp_or_requests(url, params, headers, timeout=15):
    """Fetch a URL using primp (Cloudflare bypass) with fallback to requests."""
    try:
        from primp import Client as PrimpClient
        resp = PrimpClient(impersonate="chrome_133", verify=True).get(url, params=params, headers=headers, timeout=timeout)
        return resp.status_code, resp.text
    except Exception:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        return resp.status_code, resp.text


def _format_per_store(qty_store: int, qty_branch: int) -> str:
    """Format per-store stock lines using business rules."""
    lines = []
    lines.append(_store_stock_line("Αμπελόκηποι", qty_store))
    lines.append(_store_stock_line("Παγκράτι", qty_branch))
    return "\n  ".join(lines)


def _format_total_stock(qty: int) -> str:
    """Fallback formatting when only total qty is available."""
    if qty > 2:
        return "✅ Υπάρχει στα καταστήματά μας"
    if qty in (1, 2):
        lines = [f"⚠️ Περιορισμένο απόθεμα — καλέστε για επιβεβαίωση:"]
        for store, info in STORE_INFO.items():
            lines.append(f"    📍 {store}: {info['phone']}")
        return "\n  ".join(lines)
    return "❌ Εξαντλημένο"


def check_stock(product_name: str) -> str:
    """Query the store for real-time price and per-store stock."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "el-GR,el;q=0.9,en;q=0.8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{OPENCART_URL}/",
    }

    # --- Try custom per-store API first (requires opencart_stock_api.php installed) ---
    token = _get_api_token()
    if token:
        status, text = _fetch_with_primp_or_requests(
            f"{OPENCART_URL}/index.php",
            {"route": "api/stock_locations", "api_token": token, "search": product_name},
            headers,
        )
        print(f"[check_stock:per-store] status={status} preview={text[:150]}")
        try:
            data = json.loads(text)
            if data.get("status") == "success" and data.get("products"):
                results = []
                for p in data["products"][:5]:
                    stock_str = _format_per_store(
                        int(p.get("qty_store", 0)),
                        int(p.get("qty_branch", 0)),
                    )
                    results.append(
                        f"**{p['name']}** — {p['price']}\n  {stock_str}\n  {p.get('href', '')}"
                    )
                return "\n\n".join(results)
        except Exception:
            pass  # Fall through to journal3/search

    # --- Fallback: journal3/search (total qty only) ---
    status, text = _fetch_with_primp_or_requests(
        f"{OPENCART_URL}/index.php",
        {"route": "journal3/search", "search": product_name},
        headers,
    )
    print(f"[check_stock:journal3] status={status} preview={text[:150]}")
    try:
        data = json.loads(text)
    except Exception as e:
        print(f"[check_stock] JSON error: {e}")
        return "[API_UNAVAILABLE] Could not reach store. Fall back to search_knowledge_base."

    if data.get("status") != "success" or not data.get("response"):
        return f"Δεν βρέθηκε το προϊόν '{product_name}' στο κατάστημα."

    results = []
    for p in data["response"][:5]:
        qty = int(p.get("quantity", 0))
        stock_str = _format_total_stock(qty)
        results.append(
            f"**{p.get('name', '?')}** — {p.get('price', '—')}\n  {stock_str}\n  {p.get('href', '')}"
        )
    return "\n\n".join(results)


def search_web(query: str) -> str:
    return DuckDuckGoSearchRun().run(query)


def compare_products(product_a: str, product_b: str) -> str:
    vs = _get_vectorstore()

    def fetch(name: str) -> str:
        docs = vs.similarity_search(name, k=3)
        return "\n".join(d.page_content for d in docs) if docs else "Δεν βρέθηκαν πληροφορίες."

    return f"### {product_a}\n{fetch(product_a)}\n\n### {product_b}\n{fetch(product_b)}"


def suggest_by_budget(budget: float, category: str = "") -> str:
    query = f"προϊόντα τιμή {category}".strip()
    docs = _get_vectorstore().similarity_search(query, k=12)
    found = []
    for doc in docs:
        match = re.search(r"(\d+(?:[,.]\d+)?)\s*€", doc.page_content)
        if match:
            price = float(match.group(1).replace(",", "."))
            if price <= budget:
                title = doc.metadata.get("title", "Προϊόν")
                url = doc.metadata.get("url", "")
                found.append(f"• {title}: {match.group(0)}  {url}")
    if not found:
        return f"Δεν βρέθηκαν προϊόντα εντός προϋπολογισμού {budget}€."
    return f"Προϊόντα έως {budget}€:\n" + "\n".join(found[:8])


def _call_tool(name: str, args: dict) -> str:
    if name == "search_knowledge_base":
        return search_knowledge_base(args["query"])
    if name == "check_stock":
        return check_stock(args["product_name"])
    if name == "search_web":
        return search_web(args["query"])
    if name == "compare_products":
        return compare_products(args["product_a"], args["product_b"])
    if name == "suggest_by_budget":
        return suggest_by_budget(args["budget"], args.get("category", ""))
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# OpenAI function schemas
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the microcenter.gr knowledge base for products, prices, "
                "shipping policy, returns, payment methods, and store hours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query in Greek or English"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": (
                "Check real-time stock availability and current price for a specific "
                "product from the microcenter.gr OpenCart store."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Product name or keyword to look up",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for product specifications, device compatibility, "
                "reviews, or anything not available in the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "Compare two products side by side using knowledge base information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_a": {"type": "string"},
                    "product_b": {"type": "string"},
                },
                "required": ["product_a", "product_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_by_budget",
            "description": "Suggest products within a given budget in euros, optionally filtered by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Maximum budget in euros"},
                    "category": {
                        "type": "string",
                        "description": "Optional product category, e.g. θήκη, φορτιστής, καλώδιο",
                    },
                },
                "required": ["budget"],
            },
        },
    },
]

_SYSTEM = """Είσαι ο βοηθός εξυπηρέτησης πελατών του microcenter.gr, ελληνικό e-shop τεχνολογικών αξεσουάρ.

Κανόνες:
- Απάντα στη γλώσσα που χρησιμοποιεί ο πελάτης (Ελληνικά αν γράψει Ελληνικά, Αγγλικά αν γράψει Αγγλικά).
- Για ερωτήσεις σχετικά με προϊόντα: χρησιμοποίησε ΠΑΝΤΑ και τα δύο εργαλεία: πρώτα search_knowledge_base και μετά check_stock. Και τα δύο μπορεί να έχουν διαφορετικά αποτελέσματα.
- Αν το search_knowledge_base δεν βρει αποτελέσματα, δοκίμασε ΥΠΟΧΡΕΩΤΙΚΑ το check_stock πριν απαντήσεις ότι δεν υπάρχει το προϊόν.
- Χρησιμοποίησε το check_stock για να ελέγξεις διαθεσιμότητα και τρέχουσα τιμή σε πραγματικό χρόνο.
- Χρησιμοποίησε το search_web για προδιαγραφές, συμβατότητα ή πληροφορίες που δεν υπάρχουν στη βάση.
- Χρησιμοποίησε το compare_products όταν ο πελάτης θέλει σύγκριση δύο προϊόντων.
- Χρησιμοποίησε το suggest_by_budget όταν ο πελάτης αναφέρει προϋπολογισμό.
- Όταν αναφέρεις προϊόν, πάντα συμπέριλαβε το link του.
- Παραπέμψε στο support (info@microcenter.gr) μόνο αν δεν μπορείς να βοηθήσεις με κανένα εργαλείο."""


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def ask(question: str, chat_history: list = None) -> dict:
    messages = [{"role": "system", "content": _SYSTEM}]

    if chat_history:
        for msg in chat_history:
            if hasattr(msg, "content"):
                role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append(msg)

    messages.append({"role": "user", "content": question})

    for _ in range(6):  # max tool-call iterations
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return {"answer": msg.content}

        messages.append(msg)

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _call_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return {"answer": "Δεν μπόρεσα να βρω απάντηση. Παρακαλώ επικοινωνήστε με το info@microcenter.gr"}
