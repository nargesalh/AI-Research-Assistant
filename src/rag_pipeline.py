from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# -----------------------
# 1. Load Document
# -----------------------

loader = PyPDFLoader(
    "data/papers/attention.pdf"
)

documents = loader.load()


# -----------------------
# 2. Split Document
# -----------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)


# -----------------------
# 3. Embedding
# -----------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------
# 4. Vector Database
# -----------------------

db = FAISS.from_documents(
    chunks,
    embeddings
)


# -----------------------
# 5. Retriever
# -----------------------

question = "Explain the role of Query, Key and Value in attention mechanism."


docs = db.similarity_search(
    question,
    k=3
)


context = "\n\n".join(
    [doc.page_content for doc in docs]
)


# -----------------------
# 6. LLM
# -----------------------

llm = OllamaLLM(
    model="gemma2:2b"
)


prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""


response = llm.invoke(prompt)


print(response)