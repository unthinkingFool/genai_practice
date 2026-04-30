from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text = "This is a test sentence"
# Generate the embedding for the input text -> one single vector
vector = embeddings.embed_query(text)

text=[
    "This is a test sentence",
    "This is another test sentence"
]

# Generate the embedding for the input text -> a list of vectors
vectors = embeddings.embed_documents(text)


print((vectors)) 