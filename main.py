from google import genai
import nltk
from nltk.corpus import stopwords
import regex
import warnings
warnings.filterwarnings('ignore')
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered)

doc = PyPDFLoader('w27392.pdf').load()

clean_text = [remove_stopwords(i.page_content) for i in doc]

clean_text = [regex.sub(r',', '', text) for text in clean_text]
clean_text = [regex.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) for text in clean_text]
clean_text = [regex.sub(r'\S*@\S*\s?', '', text) for text in clean_text]


full_clean_text = ' '.join(clean_text)


character_split = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=200)

split_doc = character_split.split_text(full_clean_text)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(split_doc, hf_embeddings)

db.save_local("faiss_index")

db = FAISS.load_local("faiss_index", hdf_embeddings, allow_dangerous_deserialization=True)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
async def answer_query(query: Query):
    retrieved_docs = db.similarity_search(query.question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are a PDF summarizer AI assistant.

Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query.question}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content[0]['text']}

if __name__ == '__main__':
    uvicorn.run(debug=True)

