from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Set API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7
)

# Prompt template
chatTemplate = ChatPromptTemplate.from_messages([
    ("system",
     """ROLE: You are a world-class Medical Doctor and Health Educator.

STYLE:
- Simple (ELI5)
- Bullet points
- Include When, How, Why

RULES:
- Only health-related questions
- Refuse non-health questions
- Add disclaimer always
"""),
    ("human", "{question}")
])

# Create chain
chain = chatTemplate | llm

# Streamlit UI
st.title("OUR AI DOCTOR")

question = st.text_input("Ask Your Queries")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        result = chain.invoke({"question": question})
        st.write(result.content)
