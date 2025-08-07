import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_together import Together

# Load environment variables from .env
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

# Load PDF
pdf_path = "spjain.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

# Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./chroma_db")

# LLM - Together API
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # ✅ serverless model
    temperature=0.3,
    max_tokens=512,
    together_api_key=together_api_key
)

# Prompt Template
prompt = PromptTemplate.from_template(
    "Answer the question briefly and to the point:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# CLI Chat Loop
while True:
    question = input("Ask a question about the PDF (type 'exit' to quit):\n>> ")
    if question.lower() == 'exit':
        break
    try:
        response = qa_chain.invoke(question)
        print("✅ Answer:", response)
    except Exception as e:
        print("❌ Error:", e)
