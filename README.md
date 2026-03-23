# 🤖 AI RAG Telegram Bot

A Retrieval-Augmented Generation (RAG) system built with **LangChain, FAISS, and LLaMA3 (via Groq)**, deployed as a **Telegram bot** for interactive question answering over custom documents.

---

## 🚀 Features

* 📄 Ingests PDF documents and builds a semantic search index
* 🧠 Uses **Sentence Transformers (MiniLM)** for embeddings
* 🔍 Efficient retrieval using **FAISS vector database**
* 🤖 Generates answers using **LLaMA3 via Groq API**
* 💬 Interactive Telegram bot interface
* 🧵 Supports short conversational memory
* ⚡ Lightweight and runs fully locally (except LLM API)

---

## 🧱 Tech Stack

| Component  | Technology                                 |
| ---------- | ------------------------------------------ |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector DB  | FAISS                                      |
| LLM        | LLaMA3 (via Groq API)                      |
| Framework  | LangChain                                  |
| Interface  | Telegram Bot                               |
| Language   | Python                                     |

---

## 🏗️ System Architecture

```
User (Telegram)
      ↓
Telegram Bot (/ask)
      ↓
RAG Pipeline
   ↓        ↓
FAISS     Groq (LLaMA3)
   ↓        ↓
Relevant Context + Query
      ↓
Final Answer
      ↓
Telegram Response
```

---

## 📁 Project Structure

```
├── data/                  # PDF documents
├── vectorstore/          # FAISS index (generated)
├── ingestion.py          # Builds vector DB
├── rag_pipeline.py       # RAG logic
├── bot.py                # Telegram bot
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone [<repo-url>](https://github.com/LabsVelns/AI_RAG_Telegram_bot.git)
cd AI_RAG_Telegram_bot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Environment Variables

Create a `.env` file:

```
TELEGRAM_TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key
```

---

### 5️⃣ Add Data

Place your PDF files inside:

```
data/
```

---

### 6️⃣ Build Vector Database

```bash
python ingestion.py
```

---

### 7️⃣ Run the Bot

```bash
python bot.py
```

---

## 💬 Usage

In Telegram:

```
/start
/ask What is deep learning?
/ask Explain neural networks
```

---

## 🧠 How It Works

1. Documents are split into chunks and embedded using **MiniLM**
2. Embeddings are stored in **FAISS**
3. User query is embedded and matched against stored vectors
4. Top relevant chunks are retrieved
5. Context + query is sent to **LLaMA3 (Groq)**
6. Final answer is returned to the user

---

## ⚡ APIs Used

* **Groq API**

  * Model: `llama3-8b-8192`
  * Used for response generation

---

## 📌 Notes

* Vector database must be created before running the bot
* Ensure `.env` file is configured correctly
* Do not expose API keys in public repositories

---

## 🔮 Future Improvements

* Deploy on cloud (Railway / AWS)
* Replace local embeddings with API-based embeddings
* Add web interface (Streamlit / React)

---
