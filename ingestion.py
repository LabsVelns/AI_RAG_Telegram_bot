import cohere
import fitz  # PyMuPDF
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma,FAISS
from langchain.embeddings.base import Embeddings
# from dotenv import load_dotenv

# load_dotenv()

DATA_PATH = "data/"
DB_PATH = "vectorstore/"
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Set your Cohere API key as an environment variable
# co = cohere.Client(COHERE_API_KEY)

# 🔹 Clean text
def clean_text(text):
    # remove non-ascii
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # remove phone numbers (10+ digits or with spaces)
    text = re.sub(r'\b\d{10,}\b', '', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '', text)

    # remove common noise words
    noise_words = [
        "Voluntary Contribution",
        "Whatsapp",
        "Contact",
        "Follow us",
        "linkedin.com",
        "github.com",
        "Youtube Channel",
        "Subscribe",
        "Follow me on",
        "https://www.linkedin.com/in/pedro-paulo-ribeiro-pcd-1451498/Whatsapp 55 21 999618643",
        "Blog: Fourth Industrial Revolution (22 posts)"
    ]

    for word in noise_words:
        text = text.replace(word, "")

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# 🔹 Load PDFs using PyMuPDF
def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            filepath = os.path.join(DATA_PATH, file)
            doc = fitz.open(filepath)

            for page_num, page in enumerate(doc):
                text = page.get_text()

                if not text.strip():
                    continue

                text = clean_text(text)

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file,
                            "page": page_num + 1
                        }
                    )
                )

    return documents

# 🔹 Chunking
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)

# class CohereEmbeddings(Embeddings):

#     def embed_documents(self, texts):
#         response = co.embed(
#             texts=texts,
#             model="embed-english-v3.0",
#             input_type="search_document"
#         )
#         return response.embeddings

#     def embed_query(self, text):
#         response = co.embed(
#             texts=[text],
#             model="embed-english-v3.0",
#             input_type="search_query"
#         )
#         return response.embeddings[0]

# 🔹 Create FAISS vectorstore
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    # embeddings = CohereEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)


# 🔹 Main pipeline
if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Creating vectorstore...")
    create_vectorstore(chunks)

    print("Done! Vectorstore saved.")

