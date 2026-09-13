from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------
# 1. Load PDF
# -----------------------

file_path = "data/papers/attention.pdf"

loader = PyPDFLoader(file_path)

for doc in documents:

    abstract = doc.metadata.get("description-abstract")

    if abstract:
        doc.page_content = (
            "Abstract:\n"
            + abstract
            + "\n\n"
            + doc.page_content
        )


print(f"Total pages: {len(documents)}")


# -----------------------
# 2. Split Document
# -----------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = text_splitter.split_documents(documents)


print(f"Total chunks: {len(chunks)}")


# -----------------------
# 3. Inspect Chunks
# -----------------------

for i, chunk in enumerate(chunks):

    print("\n" + "=" * 80)

    print(f"CHUNK NUMBER: {i}")

    print("Metadata:")
    print(chunk.metadata)

    print("\nContent preview:")

    print(chunk.page_content[:500])