import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

#--------------------------------Streamlit UI Setup---------------------------------------
st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬")
st.title("🎬 Movie Information Extractor")
st.write("Paste a paragraph about a movie or media content below to extract structured details.")

#--------------------------------Model Initialization-------------------------------------
# Ensure the API key is available
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("API Key not found. Please ensure GROQ_API_KEY is set in your .env file.")
    st.stop()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.7,
    max_tokens=2048
)

#------------------------------------Prompts---------------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert information extraction system specializing in analyzing unstructured text about movies or media content.

Your task is to extract only the most relevant, factual, and useful information from the given text and present it in a clean, well-structured format.

Instructions:
- Focus only on important and meaningful information.
- Ignore repetition, noise, and unnecessary storytelling details.
- Do NOT hallucinate or assume missing information.
- If any information is not clearly present, write "Not Available".
- Keep the summary concise (3–5 sentences).
- Ensure clarity, readability, and professional tone.

Format your output EXACTLY as follows:

Title:
<movie name>

Release Information:
- Release Date:
- Premiere Date:

Genre:
- <genre 1>
- <genre 2>

Director(s):
- 

Writer(s):
- 

Cast:
- Actor as Character
- Actor as Character

Summary:
<5-10 sentence professional summary>

Setting:
- Time:
- Location:

Themes:
- 
- 

Box Office:
- Budget:
- Revenue:

Awards & Nominations:
- 

Scientific / Technical Elements:
- 

Notable Elements:
- 

Key Concepts:
- 

Keywords:
- keyword1, keyword2, keyword3, ...

Text to analyze:
{context}"""
     ),

    ("human", 
     """extract the relevant information from the following text about a movie or media content and present it in the specified format:
     {paragraph}"""
     )
])

#-----------------------------------UI Components-----------------------------------------
para = st.text_area("Enter movie description/text:", height=250, placeholder="Once upon a time in a galaxy far, far away...")

if st.button("Extract Information"):
    if para.strip() == "":
        st.warning("Please provide some text first.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Prepare the messages
                final_prompt = prompt_template.format_messages(context=para, paragraph=para)
                
                # Get response from model
                response = model.invoke(final_prompt)
                
                # Display the result
                st.subheader("Extracted Information")
                st.markdown("---")
                st.text(response.content) # Using text to maintain the exact format requested
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

#---------------------------------------Footer--------------------------------------------
st.markdown("---")
st.caption("Powered by LangChain and Groq Llama-3.3")