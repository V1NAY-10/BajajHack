from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Together  # 👈 updated from deprecated path

import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Global config
pdf_path = "abc.pdf"

# Embedding model (outside to avoid reloading every time)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM setup
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.3,
    max_tokens=512
)

# Prompt setup
prompt = PromptTemplate.from_template(
    "Answer the question briefly and to the point:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
)

def get_answer_from_pdf(question: str) -> dict:
    try:
        # Step 1: Load and split PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)

        # Step 2: Store in Chroma
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./chroma_db")
        vectordb.persist()

        # Step 3: QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

        # Step 4: Get answer
        result = qa_chain.invoke(question)
        return {"question": question, "answer": result}

    except Exception as e:
        return {"question": question, "answer": f"❌ Error: {str(e)}"}
