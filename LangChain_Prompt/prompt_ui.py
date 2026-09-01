# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.7-flash")

st.header('Reasearch Tool')
user_input = st.text_input('Enter your prompt: ')
if st.button('Summarize'):
    result = model.invoke(user_input)
    st.text(result.content)
