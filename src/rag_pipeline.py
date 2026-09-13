from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# -----------------------
# 1. Load Embedding Model
# -----------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------
# 2. Load FAISS Database
# -----------------------

db = FAISS.load_local(
    "models/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


print("FAISS loaded successfully")


# -----------------------
# 3. User Question
# -----------------------

question = """
What datasets were used for evaluation?
"""


# -----------------------
# 4. Retrieve Context
# -----------------------

docs = db.similarity_search(
    question,
    k=3
)


context = "\n\n".join(
    [doc.page_content for doc in docs]
)


# -----------------------
# 5. Local LLM
# -----------------------

llm = OllamaLLM(
    model="gemma2:2b"
)


prompt = f"""
You are an AI research assistant.

Answer the question using ONLY the provided context.
If the answer is not available in the context, say:
"I don't know based on the provided document."

Be precise and include numbers exactly as they appear.

Context:
{context}

Question:
{question}

Answer:
"""

print("\n----- CONTEXT -----")
print(context)
response = llm.invoke(prompt)


print("\nAnswer:")
print(response)