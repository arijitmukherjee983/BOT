
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API keys are set
if not os.getenv("LANGCHAIN_API_KEY"):
    st.error("LANGCHAIN_API_KEY not found. Please set it in your .env file.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("LangChain Demo with Ollama API")
input_text = st.text_input("Search the topic you want")

# Run model if input is provided
if input_text:
    llm = OllamaLLM(model="mistral")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    response = chain.invoke({"question": input_text})
    st.write("Response:", response)
