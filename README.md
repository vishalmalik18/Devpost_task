# 📄 PDF AI Assistant using Gemini

An AI-powered PDF assistant that allows users to ask questions and get accurate, context-aware answers directly from PDF documents using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 📄 Extracts and processes PDF documents  
- 🔍 Semantic search using FAISS vector database  
- 🤖 AI-generated answers using Gemini API  
- ⚡ FastAPI backend for handling queries  
- ⚙️ Context-aware question answering  

---

## 🛠️ Tech Stack

- Python  
- FastAPI  
- LangChain  
- FAISS  
- Hugging Face  
- Google Gemini API  
- NLTK  

---

## ⚙️ How It Works

1. Load PDF using PyPDFLoader  
2. Clean text (remove stopwords, URLs, noise)  
3. Split text into chunks  
4. Generate embeddings using Hugging Face  
5. Store embeddings in FAISS  
6. Retrieve relevant chunks based on user query  
7. Send context + question to Gemini  
8. Generate final answer  

---

## 📦 Installation

### 1. Clone the repository

### 2. Create virtual environment
python -m venv myenv  
source myenv/Scripts/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set environment variable
set GOOGLE_API_KEY=your_api_key

---

## ▶️ Run the Project

uvicorn main:app --reload

---

## 👤 Author

**Vishal Malik**  
🔗 [LinkedIn](https://www.linkedin.com/in/vishalmalik18/)  
🔗 [GitHub](https://github.com/vishalmalik18)
