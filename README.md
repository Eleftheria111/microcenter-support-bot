# Microcenter.gr Customer Support Chatbot

An AI-powered customer support chatbot for [microcenter.gr](https://www.microcenter.gr), built using Retrieval-Augmented Generation (RAG).

## What it does
- Answers customer questions in Greek about products, prices, shipping, returns and payments
- Provides real product prices with direct links to the website
- Falls back to human support when unsure

## Tech Stack
- **LangChain** — RAG pipeline orchestration
- **OpenAI GPT-4o** — Language model
- **OpenAI text-embedding-3-small** — Text embeddings
- **ChromaDB** — Vector database
- **Playwright** — JavaScript-aware web scraping
- **BeautifulSoup** — HTML parsing
- **FastAPI** — REST API backend
- **Streamlit** — Demo chat interface

## Next Steps
- [ ] Deploy to Render.com
- [ ] JavaScript widget for website embedding
- [ ] Expand to full product catalog (33,000+ products)
- [ ] Add feedback logging
