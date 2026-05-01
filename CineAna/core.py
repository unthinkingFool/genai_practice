import os

from dotenv import load_dotenv
import streamlit as st
load_dotenv()
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

#---------------pydantic schema for the output of the model----------------------

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

#--------------------------------model initialization---------------------------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=2048
)

#------------------------------------prompts---------------------------------------
prompt=ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert information extraction system specializing in analyzing unstructured text about movies or media content.
     {format_instructions}
     """),
    ("human", "{context}")
])

parser=PydanticOutputParser(pydantic_object=MovieInfo)

para=input("give your paragraph about a movie or media content: ")

final_prompt=prompt.format_messages(format_instructions=parser.get_format_instructions(), context=para)

response=model.invoke(final_prompt)
print(response.content)