from langchain_community.document_loaders import PyPDFLoader


file_path = "data/papers/attention.pdf"

loader = PyPDFLoader(file_path)

documents = loader.load()


print(f"Number of pages: {len(documents)}")

print("\nFirst page content:")
print(documents[0].page_content[:500])

print("\nMetadata:")
print(documents[0].metadata)