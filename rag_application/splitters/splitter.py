from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader

loader = TextLoader("rag_application/data/notes.txt")
docs = loader.load()


splitter=CharacterTextSplitter(separator="",chunk_size=10, chunk_overlap=1)
chunks=splitter.split_documents(docs)
print(len(chunks))
