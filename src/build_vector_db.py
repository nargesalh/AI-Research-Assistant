from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


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


print(f"Chunks: {len(chunks)}")


# Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# Create FAISS

db = FAISS.from_documents(
    chunks,
    embeddings
)


# Save

db.save_local(
    "models/faiss_index"
)


print("FAISS saved successfully")