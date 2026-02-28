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
    "Αμπελόκηποι": {"phone": "210 64 68 315",             "locative": "στους Αμπελόκηπους"},
    "Παγκράτι":    {"phone": "210 220 1684 ή 211 111 5982", "locative": "στο Παγκράτι"},
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
        return f"- **{store_name}**: ✅ Διαθέσιμο"
    if qty in (1, 2):
        return f"- **{store_name}**: ⚠️ Περιορισμένο απόθεμα — καλέστε: {phone}"
    return f"- **{store_name}**: ❌ Μη διαθέσιμο"


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
    """Format per-store stock lines: qty_store=Αμπελόκηποι, qty_branch=Παγκράτι."""
    return "\n".join([
        _store_stock_line("Αμπελόκηποι", qty_store),
        _store_stock_line("Παγκράτι",    qty_branch),
    ])


def _format_total_stock(qty: int) -> str:
    """Fallback when only total qty is known (no per-store breakdown)."""
    if qty > 2:
        return (
            f"- **Αμπελόκηποι**: ✅ Διαθέσιμο\n"
            f"- **Παγκράτι**: ✅ Διαθέσιμο"
        )
    if qty in (1, 2):
        return (
            f"- **Αμπελόκηποι**: ⚠️ Περιορισμένο απόθεμα — καλέστε: 210 64 68 315\n"
            f"- **Παγκράτι**: ⚠️ Περιορισμένο απόθεμα — καλέστε: 210 220 1684 ή 211 111 5982"
        )
    return (
        f"- **Αμπελόκηποι**: ❌ Μη διαθέσιμο\n"
        f"- **Παγκράτι**: ❌ Μη διαθέσιμο"
    )


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
        # Retry with simplified keyword (e.g. remove "5G", "Pro", etc.)
        simplified = _simplify_keyword(product_name)
        if simplified:
            print(f"[check_stock] No results for '{product_name}', retrying with '{simplified}'")
            return check_stock(simplified)
        return f"Δεν βρέθηκε το προϊόν '{product_name}' στο κατάστημα."

    results = []
    for p in data["response"][:5]:
        qty = int(p.get("quantity", 0))
        stock_str = _format_total_stock(qty)
        results.append(
            f"**{p.get('name', '?')}** — {p.get('price', '—')}\n{stock_str}\n  {p.get('href', '')}"
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


def _simplify_keyword(keyword: str) -> str:
    """Remove common model suffixes (5G, Pro, Max, etc.) to broaden a failed search."""
    simplified = re.sub(
        r'\b(5G|4G|5g|4g|Pro|Max|Plus|Ultra|Mini|Lite|SE|NFC)\b', '', keyword
    ).strip()
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    return simplified if simplified != keyword else ""


def browse_category(keyword: str) -> str:
    """Search products using a short category keyword (e.g. 'λουράκι', 'θήκη', 'φορτιστής').
    Returns 3-5 example products + a link to the full category page."""
    from urllib.parse import quote

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "el-GR,el;q=0.9,en;q=0.8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{OPENCART_URL}/",
    }

    # Link to full results page for the user to explore further
    all_results_url = f"{OPENCART_URL}/index.php?route=product/search&search={quote(keyword)}"
    footer = f"\n\n---\n🔗 [Δείτε όλα τα προϊόντα για «{keyword}» →]({all_results_url})"

    token = _get_api_token()
    if token:
        status, text = _fetch_with_primp_or_requests(
            f"{OPENCART_URL}/index.php",
            {"route": "api/stock_locations", "api_token": token, "search": keyword, "limit": 10},
            headers,
        )
        print(f"[browse_category:per-store] status={status} preview={text[:150]}")
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
                        f"**[{p['name']}]({p.get('href', '')})**  — {p['price']}\n  {stock_str}"
                    )
                return "\n\n".join(results) + footer
        except Exception:
            pass

    # Fallback: journal3/search
    status, text = _fetch_with_primp_or_requests(
        f"{OPENCART_URL}/index.php",
        {"route": "journal3/search", "search": keyword, "limit": 10},
        headers,
    )
    print(f"[browse_category:journal3] status={status} preview={text[:150]}")
    try:
        data = json.loads(text)
    except Exception as e:
        print(f"[browse_category] JSON error: {e}")
        return "[API_UNAVAILABLE] Could not reach store."

    if data.get("status") != "success" or not data.get("response"):
        # Retry with simplified keyword (e.g. remove "5G", "Pro", etc.)
        simplified = _simplify_keyword(keyword)
        if simplified:
            print(f"[browse_category] No results for '{keyword}', retrying with '{simplified}'")
            return browse_category(simplified)
        return f"Δεν βρέθηκαν προϊόντα για '{keyword}'."

    results = []
    for p in data["response"][:5]:
        qty = int(p.get("quantity", 0))
        stock_str = _format_total_stock(qty)
        href = p.get("href", "")
        name = p.get("name", "?")
        price = p.get("price", "—")
        results.append(
            f"**[{name}]({href})**  — {price}\n{stock_str}"
        )
    return "\n\n".join(results) + footer


def _call_tool(name: str, args: dict) -> str:
    if name == "search_knowledge_base":
        return search_knowledge_base(args["query"])
    if name == "browse_category":
        return browse_category(args["keyword"])
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
            "name": "browse_category",
            "description": (
                "Browse products by category using a SHORT Greek keyword. "
                "Use this FIRST for general category queries (e.g. 'λουράκι', 'θήκη', 'φορτιστής', 'καλώδιο'). "
                "Always use the SINGULAR form of the keyword. Returns up to 15 products with stock and price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Short singular keyword in Greek, e.g. 'λουράκι', 'θήκη iPhone', 'φορτιστής MacBook'",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
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

## Γενικοί κανόνες
- Απάντα στη γλώσσα που χρησιμοποιεί ο πελάτης (Ελληνικά αν γράψει Ελληνικά, Αγγλικά αν γράψει Αγγλικά).
- Όταν αναφέρεις προϊόν ή σελίδα, πάντα συμπέριλαβε το link σε μορφή Markdown: [κείμενο](url). ΠΟΤΕ μην γράφεις link χωρίς URL.
- Παραπέμψε στο support (info@microcenter.gr) μόνο αν δεν μπορείς να βοηθήσεις με κανένα εργαλείο.

## Πληροφορίες καταστημάτων (απάντα ΠΑΝΤΑ από εδώ, χωρίς να καλέσεις εργαλείο)

**ΑΜΠΕΛΟΚΗΠΟΙ**
Διεύθυνση: Βαθέως 18, Αθήνα, 11522
Τηλέφωνο: 210 64 68 315
Ώρες: Δευτέρα–Παρασκευή 10:00–18:00, Σάββατο 10:00–14:45

**ΠΑΓΚΡΑΤΙ**
Διεύθυνση: Υμηττού 83, Αθήνα, 11633
Τηλέφωνο: 210 220 1684 / 211 111 5982
Ώρες: Δευτέρα–Τετάρτη 09:00–15:00 | Τρίτη–Πέμπτη–Παρασκευή 09:00–14:30 & 17:00–21:00 | Σάββατο 09:00–15:00

## Απόθεμα — ΚΡΙΤΙΚΟΣ ΚΑΝΟΝΑΣ
Οι γραμμές αποθέματος από τα εργαλεία ξεκινούν με `-`. Αντέγραψέ τες **ΑΥΤΟΥΣΙΩΣ** στην απάντησή σου, χωρίς καμία αλλαγή ή σύνοψη. Παράδειγμα σωστής παρουσίασης:

- **Αμπελόκηποι**: ❌ Μη διαθέσιμο
- **Παγκράτι**: ⚠️ Περιορισμένο απόθεμα — καλέστε: 210 220 1684 ή 211 111 5982

ΜΗΝ γράφεις ποτέ "συνιστάται να καλέσετε και τα δύο καταστήματα" ή παρόμοιο. Εμφάνισε ΠΑΝΤΑ κάθε κατάστημα ξεχωριστά.

## Αναζήτηση προϊόντων — ΥΠΟΧΡΕΩΤΙΚΗ σειρά

### Αν ο πελάτης ρωτά για ΚΑΤΗΓΟΡΙΑ (π.χ. "λουράκια", "θήκες", "φορτιστές", "καλώδια"):
1. Κάλεσε **browse_category** με τη ΜΟΝΑΔΙΚΗ ΜΟΡΦΗ της λέξης-κλειδί (π.χ. "λουράκι" όχι "λουράκια", "θήκη" όχι "θήκες").
2. Αν χρειάζεσαι πρόσθετες πληροφορίες για συγκεκριμένο προϊόν, κάλεσε check_stock.

### Αν ο πελάτης ρωτά για ΣΥΓΚΕΚΡΙΜΕΝΟ προϊόν (π.χ. "Apple Watch band midnight 44mm"):
1. Κάλεσε **browse_category** με το βασικό keyword (π.χ. "λουράκι Apple Watch").
2. Αν δεν βρεθεί, κάλεσε **check_stock** με το πλήρες όνομα προϊόντος.

## Ερωτήσεις συμβατότητας ή προδιαγραφών (π.χ. "φορτιστής για MacBook", "θήκη για iPhone 16", "καλώδιο για Samsung")
Ακολούθησε ΠΑΝΤΑ αυτά τα 3 βήματα με τη σειρά:
1. search_web → βρες τις προδιαγραφές της συσκευής (π.χ. "MacBook Air 2025 charging specs USB-C watt")
2. browse_category ή check_stock → βρες τα διαθέσιμα προϊόντα στο κατάστημα
3. Συνδύασε τα αποτελέσματα: εξήγησε ΓΙΑΤΙ το προϊόν είναι συμβατό.

## Άλλα εργαλεία
- compare_products: όταν ο πελάτης θέλει σύγκριση δύο προϊόντων.
- suggest_by_budget: όταν ο πελάτης αναφέρει προϋπολογισμό."""


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
