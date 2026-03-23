from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
import requests
from dotenv import load_dotenv
import os
import cohere

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set your Groq API key as an environment variable
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Set your Cohere API key as an environment variable
co = cohere.Client(COHERE_API_KEY)

DB_PATH = "vectorstore/"
query_cache = {}  # Simple in-memory cache for query results

# 🔹 Embed query using Cohere api
def embed_query(query):
    response = co.embed(
        texts=[query],
        model="embed-english-light-v3.0"
    )
    return response.embeddings[0]

# 🔹 Load vector DB
def load_db():
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="all-MiniLM-L6-v2"
    # )

    db = FAISS.load_local(DB_PATH, allow_dangerous_deserialization=True) #, embeddings
    return db


# 🔹 Retrieve context
def retrieve_docs(db, query, k=3):
    # results = db.similarity_search(query, k=k)

    # retriever = db.as_retriever(
    # search_type="mmr",
    # search_kwargs={"k": 3, "fetch_k": 10}
    # )
    query_vector = embed_query(query)
    results = db.similarity_search_by_vector(query_vector, k=3)

    # results = retriever.invoke(query)
    return results

# 🔹 Build prompt
# def build_prompt(query, docs, history=None):
#     context = "\n\n".join([doc.page_content for doc in docs])

#     prompt = f"""
# You are an AI assistant. Answer the question based ONLY on the context below.

# Context:
# {context}

# Question:
# {query}

# Answer clearly and concisely.
# """
#     return prompt

def build_prompt(query, docs, history=None):
    context = "\n\n".join([doc.page_content[:300] for doc in docs])

    history_text = ""
    if history:
        history_text = "\n".join(history)

    prompt = f"""
You are a helpful AI assistant.

Previous conversation:
{history_text}

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt


# # 🔹 Call Ollama (LLaMA3)
# def call_ollama(prompt):
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "llama3.2:latest",
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     return response.json()["response"]

def call_llm(prompt):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]


# 🔹 Full pipeline
def ask(query,history=None):
    # 🔹 Check cache
    query_key = query.lower().strip()
    if query_key in query_cache:
        return query_cache[query_key]
    
    db = load_db()

    # docs = retrieve_docs(db, query)
    query_vector = embed_query(query)
    docs = db.similarity_search_by_vector(query_vector, k=3)

    prompt = build_prompt(query, docs, history=history)

    # answer = call_ollama(prompt)
    answer = call_llm(prompt)
    # 🔹 Store in cache
    
    query_cache[query_key] = (answer, docs)

    return answer, docs


# 🔹 Test
if __name__ == "__main__":
    query = input("Ask something: ")

    answer, docs = ask(query)

    print("\n--- ANSWER ---\n")
    print(answer)

    print("\n--- SOURCES ---\n")
    for doc in docs:
        print(doc.metadata)