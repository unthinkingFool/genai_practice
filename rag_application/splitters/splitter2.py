from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("rag_application/data/GRU.pdf")
docs = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=1)

chunks = text_splitter.split_documents(docs)
print(chunks[0].page_content)