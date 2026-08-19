import langchain_community.llms 
import ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


prompt = ChatPromptTemplate(
    [
        ("system", "Tu hi mera sabse acha AI bot hai. Please respond  mein!!"),
        ("user", "Question: {question}")
    ]
)

st.title("Aloo Chaat JiPiTi!!")
input_text = st.text_input("Jo puchna hai yaha puch !!!")

llm = OllamaLLM(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
if input_text:
    st.write(chain.invoke({"question": input_text}))