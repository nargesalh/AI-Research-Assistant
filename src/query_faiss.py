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


results = db.similarity_search_with_score(
    question,
    k=3
)


for i, (doc, score) in enumerate(results):

    print(f"\n--- Result {i+1} ---")

    print("Score:", score)

    print(doc.page_content[:500])

    print("\nMetadata:")
    print(doc.metadata)