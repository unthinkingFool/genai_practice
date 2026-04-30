from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
load_dotenv()

# keeping the chat history in a list
message=[]

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=2048
)

while(True):
    question = input("you: ")
    message.append(question)
    if question.lower() == "exit":
        break
    response=model.invoke(message)
    message.append(response.content)
    print("bot: ",response.content)