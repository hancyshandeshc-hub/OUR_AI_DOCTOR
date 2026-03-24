from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

# Prompt
chatTemplate = ChatPromptTemplate.from_messages([
    ("system", """ROLE: You are a world-class Medical Doctor and Health Educator.

- Use simple explanations (ELI5)
- Use bullet points
- Include When, How, Why
- Only answer health-related questions
- Always include disclaimer
"""),
    ("human", "{question}")
])

# UI
st.title("OUR AI DOCTOR")
question = st.text_input("Ask Your Queries")

# Button logic FIXED
if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Consulting AI Doctor..."):
            prompt = chatTemplate.invoke({"question": question})
            result = model.invoke(prompt)
            st.write(result.content)
