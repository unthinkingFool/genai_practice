from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("rag_application/data/GRU.pdf")
docs = loader.load()

print(docs[0].page_content)