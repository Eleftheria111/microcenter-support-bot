import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

client = OpenAI()

# --- Vectorstore (loaded once) ---

_vectorstore = None

def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        _vectorstore = FAISS.load_local(
            'data/faiss', embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore


# --- Tool implementations ---

def search_knowledge_base(query: str) -> str:
    docs = _get_vectorstore().similarity_search(query, k=6)
    if not docs:
        return "Δεν βρέθηκαν αποτελέσματα στη βάση γνώσεων."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title', '')}]\nURL: {d.metadata.get('url', '')}\n{d.page_content}"
        for d in docs
    )


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
        match = re.search(r'(\d+(?:[,.]\d+)?)\s*€', doc.page_content)
        if match:
            price = float(match.group(1).replace(',', '.'))
            if price <= budget:
                title = doc.metadata.get('title', 'Προϊόν')
                url = doc.metadata.get('url', '')
                found.append(f"• {title}: {match.group(0)}  {url}")
    if not found:
        return f"Δεν βρέθηκαν προϊόντα εντός προϋπολογισμού {budget}€."
    return f"Προϊόντα έως {budget}€:\n" + "\n".join(found[:8])


def _call_tool(name: str, args: dict) -> str:
    if name == "search_knowledge_base":
        return search_knowledge_base(args["query"])
    if name == "search_web":
        return search_web(args["query"])
    if name == "compare_products":
        return compare_products(args["product_a"], args["product_b"])
    if name == "suggest_by_budget":
        return suggest_by_budget(args["budget"], args.get("category", ""))
    return "Unknown tool."


# --- OpenAI function schemas ---

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the microcenter.gr knowledge base for products, prices, shipping, returns, payment methods, and store hours.",
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
            "name": "search_web",
            "description": "Search the web for product specifications, device compatibility, or anything not in the knowledge base.",
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
            "description": "Suggest products within a given budget in euros. Optionally filter by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Maximum budget in euros"},
                    "category": {"type": "string", "description": "Optional product category, e.g. θήκη, φορτιστής, καλώδιο"},
                },
                "required": ["budget"],
            },
        },
    },
]

_SYSTEM = """Είσαι ο βοηθός εξυπηρέτησης πελατών του microcenter.gr, ελληνικό e-shop τεχνολογικών αξεσουάρ.

Κανόνες:
- Απάντα στη γλώσσα που χρησιμοποιεί ο πελάτης (Ελληνικά αν γράψει Ελληνικά, Αγγλικά αν γράψει Αγγλικά).
- Χρησιμοποίησε πρώτα το search_knowledge_base για προϊόντα και πολιτικές του καταστήματος.
- Χρησιμοποίησε το search_web για προδιαγραφές, συμβατότητα ή πληροφορίες που δεν υπάρχουν στη βάση.
- Χρησιμοποίησε το compare_products όταν ο πελάτης θέλει σύγκριση δύο προϊόντων.
- Χρησιμοποίησε το suggest_by_budget όταν ο πελάτης αναφέρει προϋπολογισμό.
- Παραπέμψε στο support (info@microcenter.gr) μόνο αν δεν μπορείς να βοηθήσεις με κανένα εργαλείο."""


# --- Agent loop ---

def ask(question: str, chat_history: list = None) -> dict:
    messages = [{"role": "system", "content": _SYSTEM}]

    if chat_history:
        for msg in chat_history:
            if hasattr(msg, 'content'):
                role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append(msg)

    messages.append({"role": "user", "content": question})

    for _ in range(5):  # max tool-call iterations
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

        # Append assistant message with tool calls
        messages.append(msg)

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _call_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return {"answer": "Δεν μπόρεσα να βρω απάντηση. Παρακαλώ επικοινωνήστε με το info@microcenter.gr"}


# Kept for backwards compatibility with api.py / any callers
def get_agent():
    return None
