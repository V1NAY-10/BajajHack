import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_together import Together

# Load API key from .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Load PDF
pdf_path = "spjain.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

# Embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in ChromaDB
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./chroma_db")

# LLM setup
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.3,
    together_api_key=TOGETHER_API_KEY
)

# Prompt
prompt = PromptTemplate.from_template(
    "Answer the question briefly and to the point ONLY using the provided context.\n"
    "If the answer is not in the context, say 'Not available in PDF'.\n\n"
    "Context: {context}\n\nQuestion: {question}\nAnswer:"
)

# Retrieval QA
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True  # ✅ so we can check if sources exist
)

# Chat loop
while True:
    question = input("Ask a question about the PDF (type 'exit' to quit):\n>> ")
    if question.lower() == "exit":
        break

    # Retrieve context first
    docs = retriever.get_relevant_documents(question)
    if not docs:
        print("📄 Not available in PDF")
        continue

    # Run LLM if we have context
    try:
        response = qa_chain.invoke({"query": question})
        print("✅ Answer:", response["result"])
    except Exception as e:
        print("❌ Error:", e)
