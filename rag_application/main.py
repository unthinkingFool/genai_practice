from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()

#-------------------model-------------------

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=2048
)

#-------------------embeddings-------------------

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#-------------------loading vectorstore-------------------

vectorstore=Chroma(embedding_function=embeddings,persist_directory="chroma-db")

#-------------------retriever-------------------

retriever = vectorstore.as_retriever(search_kwargs={"k": 3,"fetch_k":10,"lambda_mult":0.5},search_type="mmr")

#-------------------prompt template-------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

#-------------------query-------------------
print("============================================")
print("press 0 to exit")
while True:
    query = input("Enter your question: ")
    if query == "0":
        print("Exiting...")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt= prompt.format(context=context, question=query)

    response = model.invoke(final_prompt)

    print("\nAnswer:\n")
    print(response.content)
    print("============================================")
