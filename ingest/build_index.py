import json, os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# Load scraped pages
docs = []
with open('data/processed/pages.jsonl') as f:
    for line in f:
        page = json.loads(line)
        docs.append(Document(
            page_content=page['text'],
            metadata={'url': page['url'], 'title': page['title']}
        ))

print(f'Loaded {len(docs)} pages')

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f'Split into {len(chunks)} chunks')

# Create embeddings and store in ChromaDB
print('Creating embeddings...')
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory='data/chroma'
)
print('Index built and saved to data/chroma/')