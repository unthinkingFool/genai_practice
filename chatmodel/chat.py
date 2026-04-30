from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
load_dotenv()



model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=2048
)

response=model.invoke("What is the capital of France?")
print(response.content)