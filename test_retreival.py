from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "vectorstore/"

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.load_local(DB_PATH, embeddings,allow_dangerous_deserialization=True)


query = "What is deep learning?"

results = db.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results):
    print(f"\n--- RESULT {i+1} ---")
    print(f"Score: {score}")
    print(doc.page_content[:500])  # Print the first 500 characters of the content
    print("SOURCE:", doc.metadata)