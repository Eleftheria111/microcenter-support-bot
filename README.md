# 🤖 Microcenter.gr Customer Support Chatbot

An AI-powered customer support chatbot for [microcenter.gr](https://www.microcenter.gr), built using Retrieval-Augmented Generation (RAG).

## 🎯 What it does
- Answers customer questions in Greek about products, prices, shipping, returns and payments
- Provides real product prices with direct links to the website
- Knows store locations and opening hours
- Falls back to human support when unsure

## 🛠️ Tech Stack
- **LangChain** — RAG pipeline orchestration
- **OpenAI GPT-4o** — Language model
- **OpenAI text-embedding-3-small** — Text embeddings
- **ChromaDB** — Vector database
- **Playwright** — JavaScript-aware web scraping
- **BeautifulSoup** — HTML parsing
- **FastAPI** — REST API backend
- **Streamlit** — Demo chat interface

## 🏗️ Architecture
```
Website Pages → Scraper → Chunks → Embeddings → ChromaDB
                                                     ↓
Customer Question → Retriever → Top-K Chunks → GPT-4o → Answer
```

## 🚀 How to run

### 1. Clone and setup
```bash
git clone https://github.com/Eleftheria111/microcenter-support-bot.git
cd microcenter-support-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 3. Scrape and build index
```bash
python ingest/fetch_pages.py
python ingest/fetch_products.py
python ingest/build_index.py
```

### 4. Run the demo
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure
```
microcenter-bot/
  app/
    api.py        # FastAPI backend
    rag.py        # RAG pipeline
  ingest/
    fetch_pages.py      # Scrape support pages
    fetch_products.py   # Scrape products with Playwright
    build_index.py      # Build ChromaDB vector index
  data/
    urls.txt            # Support page URLs
    product_urls.txt    # Product URLs
  streamlit_app.py      # Demo UI
  README.md
```

## 💡 Key Features
- **No hallucinations** — answers only from scraped website content
- **Bilingual** — works in Greek and English
- **Source citations** — every answer includes product links
- **Scalable** — easily add more products or pages

## 📊 Current Knowledge Base
- 7 support pages (shipping, returns, payments, contact, etc.)
- 50 Samsung phone cases with real prices
- Store locations and opening hours

## 🔮 Next Steps
- [ ] Deploy to Render.com
- [ ] JavaScript widget for website embedding
- [ ] Expand to full product catalog (33,000+ products)
- [ ] Add feedback logging
