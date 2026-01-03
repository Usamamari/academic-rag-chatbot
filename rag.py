import os
import requests
from pypdf import PdfReader
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LOAD .env FILE
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_PATH = "data/research_papers"
TOP_K = 3

documents = []
vectorizer = None
doc_vectors = None


def load_documents():
    global documents, vectorizer, doc_vectors

    texts = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_PATH, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)

    if not texts:
        raise ValueError("❌ No research papers found.")

    documents = texts
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)


def generate_answer(context, question):
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY missing. Check .env file."

    prompt = f"""
You are an academic research assistant.
Answer clearly and in detail using ONLY the context.

Context:
{context}

Question:
{question}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "Academic Assistant"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4
        },
        timeout=30
    )

    data = response.json()
    return data["choices"][0]["message"]["content"]


def ask(question: str) -> str:
    q = question.lower()

    # ---- DEVELOPER INFO ----
    if "usama" in q or "developer" in q:
        return (
            "Usama is a BS Computer Science student and the developer of this "
            "Academic Research Assistant, built as part of his Final Year Project (FYP)."
        )

    # ---- SYSTEM INFO ----
    if "llm" in q or "model" in q:
        return (
            "This system uses a Retrieval-Augmented Generation (RAG) architecture. "
            "Relevant research papers are retrieved using TF-IDF similarity, and "
            "answers are generated using the LLaMA 3.1 model via Groq API."
        )

    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    top_idx = scores.argsort()[-TOP_K:][::-1]

    context = "\n".join([documents[i] for i in top_idx])
    return generate_answer(context, question)
