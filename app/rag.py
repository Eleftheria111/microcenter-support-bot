from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROMPT_TEMPLATE = """
You are a helpful customer support assistant for microcenter.gr,
a technology products company in Greece.
The knowledge base is in Greek. The customer may ask in any language.
Answer using ONLY the information below.
If the answer is not in the context, say:
"I'm not sure about that. Please contact our support team at microcenter.gr"
Always answer in the same language the customer used.

Context:
{context}

Customer Question: {question}
Answer:
"""

def translate_to_greek(question: str, llm) -> str:
    result = llm.invoke(f"Translate this to Greek, return ONLY the translation: {question}")
    return result.content

def get_qa_chain():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.load_local('data/faiss', embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 8})
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=PROMPT_TEMPLATE
    )
    return retriever, llm, prompt

def ask(question: str, chain=None):
    retriever, llm, prompt = get_qa_chain()
    
    # Translate question to Greek for better retrieval
    greek_question = translate_to_greek(question, llm)
    
    # Retrieve relevant chunks using Greek question
    docs = retriever.invoke(greek_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Answer in original language
    final_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(final_prompt)
    
    return {
        'answer': answer.content,
        'sources': [doc.metadata.get('url', '') for doc in docs]
    }
