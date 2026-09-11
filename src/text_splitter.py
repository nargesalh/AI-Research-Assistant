from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


file_path = "data/papers/attention.pdf"


loader = PyPDFLoader(file_path)

documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = text_splitter.split_documents(documents)


print(f"Number of chunks: {len(chunks)}")


print("\nFirst chunk:")
print(chunks[0].page_content[:500])


print("\nMetadata:")
print(chunks[0].metadata)