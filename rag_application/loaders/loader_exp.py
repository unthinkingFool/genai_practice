from langchain_community.document_loaders import TextLoader

loader = TextLoader("rag_application/data/notes.txt")
docs = loader.load()

print(docs[0])