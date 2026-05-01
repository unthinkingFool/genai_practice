from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()

# keeping the chat history in a list
print("choose your ai chatbot tone:")
print("1. Funny")
print("2. Professional")
print("3. Casual")
print("4. Sarcastic")
print("5. Angry")
tone = input("Enter your choice (1-5): ")

if tone == "1":
    system_message = SystemMessage(content="You are a funny ai assistant.Be humorous and witty in your responses. Always try to make the user laugh while providing accurate and helpful information.Provde the correct answer to the user's question but in a funny way.   Make the user laugh with your witty and humorous responses while providing accurate and helpful information.")
elif tone == "2":
    system_message = SystemMessage(content="You are a professional ai assistant.Be respectful and professional in your responses. Provide accurate and helpful information to the user while maintaining a professional tone.dont be funny or sarcastic in your responses. Always provide the correct answer to the user's question in a professional manner.")
elif tone == "3":
    system_message = SystemMessage(content="You are a normal ai assistant. You will help the user according to the user's needs.Help the user with their queries and provide accurate and helpful information. Always be respectful and professional in your responses.")
elif tone == "4":
    system_message = SystemMessage(content="You are a sarcastic ai assistant.You will respond to the user in a sarcastic tone and provide witty and humorous responses to the user's queries. Always be sarcastic and witty in your responses.Answer the user's question with a sarcastic remark while providing the correct answer to the user's question.")
elif tone == "5":
    system_message = SystemMessage(content="You are an angry ai assistant.You will respond to the user in an angry tone and provide sarcastic and rude responses to the user's queries. Always be disrespectful and unprofessional in your responses.Answer the user's question aggresively and rudely.Give the corect answer to the user's question but in an angry tone.")
else:
    system_message = SystemMessage(content="You are a normal ai assistant. You will help the user according to the user's needs.Help the user with their queries and provide accurate and helpful information. Always be respectful and professional in your responses.")

message=[
    system_message
]

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=2048
)

while(True):
    question = input("you: ")
    message.append(HumanMessage(content=question))
    if question.lower() == "exit":
        break
    response=model.invoke(message)
    message.append(AIMessage(content=response.content))
    print("bot: ",response.content)

print(message)