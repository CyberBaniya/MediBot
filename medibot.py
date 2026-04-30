import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="MediBot", layout="wide")

# Title
st.title("🩺 Medical Chatbot (RAG)")
st.write("Ask your medical questions based on documents")

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Load embedding + DB (load only once)
@st.cache_resource
def load_db():
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embedding,
        allow_dangerous_deserialization=True
    )
    return db

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY
)

# Prompt
prompt_template = PromptTemplate.from_template("""
You are a helpful medical assistant.

Use the context to answer the question.
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""")

# Chat input (ONLY ONE)
user_input = st.chat_input("Describe your problem...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Retrieve docs
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Final prompt
    final_prompt = prompt_template.format(
        context=context,
        question=user_input
    )

    # LLM response
    response = llm.invoke(final_prompt).content

    # Show response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })