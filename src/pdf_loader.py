from langchain_community.document_loaders import PyPDFLoader


file_path = "data/papers/attention.pdf"


loader = PyPDFLoader(file_path)

documents = loader.load()


for doc in documents:

    abstract = doc.metadata.get("description-abstract")

    if abstract:
        doc.page_content = (
            "Abstract:\n"
            + abstract
            + "\n\n"
            + doc.page_content
        )


print(f"Number of pages: {len(documents)}")


print("\nFirst page content:")
print(documents[0].page_content[:700])


print("\nMetadata:")
print(documents[0].metadata)