import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

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


# --- Tools ---

@tool
def search_knowledge_base(query: str) -> str:
    """Search the microcenter.gr knowledge base for products, prices, shipping,
    returns, payment methods, and store hours."""
    docs = _get_vectorstore().similarity_search(query, k=6)
    if not docs:
        return "Δεν βρέθηκαν αποτελέσματα στη βάση γνώσεων."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title', '')}]\n{d.page_content}" for d in docs
    )


@tool
def search_web(query: str) -> str:
    """Search the web for product specifications, device compatibility, reviews,
    or anything not available in the knowledge base."""
    return DuckDuckGoSearchRun().run(query)


@tool
def compare_products(product_a: str, product_b: str) -> str:
    """Compare two products side by side using knowledge base information."""
    vs = _get_vectorstore()

    def fetch(name: str) -> str:
        docs = vs.similarity_search(name, k=3)
        return "\n".join(d.page_content for d in docs) if docs else "Δεν βρέθηκαν πληροφορίες."

    return f"### {product_a}\n{fetch(product_a)}\n\n### {product_b}\n{fetch(product_b)}"


@tool
def suggest_by_budget(budget: float, category: str = "") -> str:
    """Suggest products within a given budget (in euros).
    Optionally filter by category, e.g. 'θήκη', 'φορτιστής', 'καλώδιο'."""
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


# --- Agent ---

_SYSTEM = """Είσαι ο βοηθός εξυπηρέτησης πελατών του microcenter.gr, ελληνικό e-shop τεχνολογικών αξεσουάρ.

Κανόνες:
- Απάντα στη γλώσσα που χρησιμοποιεί ο πελάτης (Ελληνικά αν γράψει Ελληνικά, Αγγλικά αν γράψει Αγγλικά).
- Χρησιμοποίησε πρώτα το search_knowledge_base για προϊόντα και πολιτικές του καταστήματος.
- Χρησιμοποίησε το search_web για προδιαγραφές, συμβατότητα ή πληροφορίες που δεν υπάρχουν στη βάση.
- Χρησιμοποίησε το compare_products όταν ο πελάτης θέλει σύγκριση δύο προϊόντων.
- Χρησιμοποίησε το suggest_by_budget όταν ο πελάτης αναφέρει προϋπολογισμό.
- Παραπέμψε στο support (info@microcenter.gr) μόνο αν δεν μπορείς να βοηθήσεις με κανένα εργαλείο."""

_agent_executor = None


def get_agent() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        llm = ChatOpenAI(model='gpt-4o', temperature=0)
        tools = [search_knowledge_base, search_web, compare_products, suggest_by_budget]
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        _agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, max_iterations=5
        )
    return _agent_executor


def ask(question: str, chat_history: list = None) -> dict:
    result = get_agent().invoke({
        "input": question,
        "chat_history": chat_history or [],
    })
    return {"answer": result["output"]}
