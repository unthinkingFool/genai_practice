import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint



llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
   
)
model = ChatHuggingFace(llm=llm)

print(model.invoke("What is the capital of France?").content)