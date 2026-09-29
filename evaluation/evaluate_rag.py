import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
import json
from nltk.stem import PorterStemmer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL, TOP_K
from langchain_ollama import OllamaLLM
import re
from sentence_transformers import util


def normalize(text):
    text = text.lower()

    replacements = {
        "-": " ",
        "_": " "
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    words = text.split()

    synonym_map = {
        "queries": "query",
        "querie": "query",
        "keys": "key",
        "values": "value",
        "heads": "head"
    }

    words = [
        synonym_map.get(word, word)
        for word in words
    ]
    stemmer = PorterStemmer()

    processed_words = []

    for word in words:
        word = stemmer.stem(word)
        processed_words.append(word)

    return " ".join(processed_words)

# -----------------------
# Load questions
# -----------------------

with open(
    "evaluation/questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


# -----------------------
# Load embedding
# -----------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# -----------------------
# Load FAISS
# -----------------------

db = FAISS.load_local(
    "models/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


print("FAISS loaded")


# -----------------------
# Load LLM
# -----------------------

llm = OllamaLLM(
    model="gemma2:2b"
)

total_semantic_score = 0
total_score = 0
total_possible = 0


# -----------------------
# Evaluation loop
# -----------------------

for item in questions:

    question = item["question"]
    keywords = item["expected_keywords"]
    print("\nDEBUG KEYWORDS:")
    print(keywords)

    docs = db.max_marginal_relevance_search(
    question,
    k=TOP_K
)


    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    prompt = f"""
Answer the question using ONLY the provided context.

Rules:
- Extract all relevant information from the context.
- If the answer asks for multiple items, return all items.
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


    answer = llm.invoke(prompt)
    expected_answer = item["expected_answer"]

    answer_embedding = embeddings.embed_query(answer)

    expected_embedding = embeddings.embed_query(expected_answer)

    semantic_score = util.cos_sim(
        answer_embedding,
        expected_embedding
    ).item()


    print("\n==============================")
    print("Question:")
    print(question)

    print("\nAnswer:")
    print(answer)


    score = 0

    print("\nKeywords:")

    normalized_answer = normalize(answer)

    print("\nNormalized Answer:")
    print(normalized_answer)


    for keyword in keywords:

        normalized_keyword = normalize(keyword)

        print(
            "Checking:",
            normalized_keyword
        )

        if normalized_keyword in normalized_answer:
            print("✓", keyword)
            score += 1
        else:
            print("✗", keyword)


    print(
        f"\nScore: {score}/{len(keywords)}"
    )

    print(
        f"Semantic Similarity: {semantic_score:.2f}"
    )

    total_semantic_score += semantic_score
    total_score += score
    total_possible += len(keywords)



print("\n==============================")
print("FINAL RESULT")

print(
    f"Score: {total_score}/{total_possible}"
)

print(
    f"Accuracy: {(total_score/total_possible)*100:.2f}%"
)

print(
    f"Average Semantic Similarity: {total_semantic_score / len(questions):.2f}"
)