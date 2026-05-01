from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("rag_application/data/GRU.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    
)
texts = text_splitter.split_documents(docs)
print(texts[0])
