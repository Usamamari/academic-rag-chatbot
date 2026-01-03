import streamlit as st
from rag import load_documents, ask

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📘",
    layout="wide"
)

@st.cache_resource
def init():
    load_documents()

init()

# ---------- SIDEBAR ----------
st.sidebar.title("📌 Project Overview")
st.sidebar.write("""
**Academic Research Assistant**

A Retrieval-Augmented Generation (RAG) based chatbot
that answers questions strictly from uploaded research papers.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed By**")
st.sidebar.markdown("""
**Usama**  
BS Computer Science  
Final Year Project
""")

# ---------- MAIN ----------
st.title("📘 Academic Research Assistant")
st.write("Ask questions related to your uploaded research papers.")

question = st.text_input("❓ Ask your academic question")

if question:
    with st.spinner("Analyzing documents..."):
        answer = ask(question)

    st.markdown("### 📖 Answer")
    st.write(answer)

# ---------- FOOTER ----------
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray;">
    Developed by <b>Usama</b> | Final Year Project
    </div>
    """,
    unsafe_allow_html=True
)
