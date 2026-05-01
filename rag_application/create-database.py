#load pdf
#split into chunks
#embed the chunks
#store the chunks and embeddings in a vector database

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()

#-------------------load pdf-------------------

loader=PyPDFLoader("rag_application/data/deeplearning.pdf")
docs=loader.load()

#-------------------split into chunks-------------------

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks=splitter.split_documents(docs)

#-------------------embed the chunks-------------------

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#-------------------store the chunks and embeddings in a vector database-------------------

vectordb=Chroma.from_documents(documents=chunks,embedding=embeddings,persist_directory="chroma-db")

