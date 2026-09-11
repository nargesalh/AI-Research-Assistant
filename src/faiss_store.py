from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# 1. Load PDF

file_path = "data/papers/attention.pdf"

loader = PyPDFLoader(file_path)

documents = loader.load()


# 2. Split text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


print(f"Chunks: {len(chunks)}")


# 3. Create Embedding Model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create Vector Database

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


print("FAISS database created")