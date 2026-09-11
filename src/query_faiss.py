from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load PDF

loader = PyPDFLoader(
    "data/papers/attention.pdf"
)

documents = loader.load()


# Split

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)


# Embedding

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS

db = FAISS.from_documents(
    chunks,
    embeddings
)


# Query

question = "What is attention mechanism?"


results = db.similarity_search(
    question,
    k=3
)


for i, doc in enumerate(results):
    print("\n--- Result", i+1, "---")
    print(doc.page_content[:500])