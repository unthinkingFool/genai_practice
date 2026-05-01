import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Load environment variables
load_dotenv()

#-------------------pydantic schema for the output of the model----------------------
class MovieInfo(BaseModel):
    title: str = Field(..., description="The title of the movie or media content.")
    release_information: dict = Field(..., description="Release information including release date and premiere date.")
    genre: List[str] = Field(..., description="List of genres associated with the movie.")
    directors: List[str] = Field(..., description="List of directors of the movie.")
    writers: List[str] = Field(..., description="List of writers of the movie.")
    cast: List[str] = Field(..., description="List of main cast members and their characters.")
    summary: str = Field(..., description="A concise professional summary of the movie.")
    rating: Optional[str] = Field(None, description="The movie's rating (e.g., PG-13, R).")
    setting: dict = Field(..., description="Setting information including time and location.")
    themes: List[str] = Field(..., description="List of themes explored in the movie.")
    box_office: dict = Field(..., description="Box office information including budget and revenue.")
    awards_nominations: List[str] = Field(..., description="List of awards and nominations received by the movie.")
    scientific_technical_elements: List[str] = Field(..., description="List of scientific or technical elements featured in the movie.")
    notable_elements: List[str] = Field(..., description="List of notable elements such as unique storytelling techniques or visual styles.")
    key_concepts: List[str] = Field(..., description="List of key concepts or ideas explored in the movie.")
    keywords: List[str] = Field(..., description="List of relevant keywords associated with the movie.")

#--------------------------------Streamlit Setup-------------------------------------------
st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬")
st.title("🎬 Movie Information Extractor")
st.markdown("Extract structured data from unstructured movie descriptions using AI.")

#--------------------------------model initialization---------------------------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Error: GROQ_API_KEY not found in environment variables.")
    st.stop()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.7,
    max_tokens=2048
)

#------------------------------------prompts---------------------------------------
parser = PydanticOutputParser(pydantic_object=MovieInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert information extraction system specializing in analyzing unstructured text about movies or media content.
     {format_instructions}
     """),
    ("human", "{context}")
])

#------------------------------------UI Logic---------------------------------------
# Input section
para = st.text_area("Give your paragraph about a movie or media content:", height=200)

if st.button("Extract Data"):
    if not para.strip():
        st.warning("Please enter some text before extracting.")
    else:
        with st.spinner("Analyzing text and extracting data..."):
            try:
                # Format the prompt
                final_prompt = prompt.format_messages(
                    format_instructions=parser.get_format_instructions(), 
                    context=para
                )
                
                # Invoke the model
                response = model.invoke(final_prompt)
                
                # Parse the output
                parsed_data = parser.parse(response.content)
                
                # Display Results
                st.subheader(f"Results: {parsed_data.title}")
                
                # Display as a clean JSON-like view (as requested to keep features same)
                st.json(parsed_data.dict())
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

#------------------------------------Footer-----------------------------------------
st.markdown("---")
st.caption("Powered by LangChain, Groq, and Streamlit")