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

docs = db.max_marginal_relevance_search(
    question,
    k=8
)


context = ""

for i, doc in enumerate(docs):

    context += f"""
--- Document Chunk {i+1} ---

Source:
{doc.metadata.get('source')}

Page:
{doc.metadata.get('page')}

Content:
{doc.page_content}

"""


# -----------------------
# 5. Local LLM
# -----------------------

llm = OllamaLLM(
    model="gemma2:2b"
)


prompt = f"""
Answer the question using ONLY the provided context.

Rules:
- Extract all relevant information from the context.
- If the question asks for multiple items, return all items.
- Do not return only one example.
- Do not use outside knowledge.
- If the answer is not available in the context, say:
"I don't know based on the provided document."

Be concise and accurate.

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