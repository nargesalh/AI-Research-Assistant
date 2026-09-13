import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import re


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

    processed_words = []

    for word in words:
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]

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
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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

    docs = db.similarity_search(
        question,
        k=5
    )


    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    prompt = f"""
You are an AI research assistant.

Answer only using the context.

Context:
{context}


Question:
{question}

Answer:
"""


    answer = llm.invoke(prompt)


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