from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_together import Together
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("❌ TOGETHER_API_KEY not found in .env file")

# 📁 Hardcoded file path (change as needed)
FILE_PATH = "research.docx"  # Can be .pdf, .docx, .eml

# 1. Load the file
print(f"📄 Loading file: {FILE_PATH}")
if FILE_PATH.endswith(".pdf"):
    loader = PyPDFLoader(FILE_PATH)
elif FILE_PATH.endswith(".docx"):
    loader = Docx2txtLoader(FILE_PATH)
elif FILE_PATH.endswith(".eml"):
    loader = UnstructuredEmailLoader(FILE_PATH)
else:
    raise ValueError("❌ Unsupported file format. Use .pdf, .docx, or .eml")

documents = loader.load()
print(f"✅ Loaded {len(documents)} documents.")

# 2. Split content
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Embed and store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding=embedding, persist_directory="./chroma_db")

# 4. Prompt
prompt = PromptTemplate.from_template(
    "Answer the question using only the information in the context below. "
    "If the answer is not available in the context, say 'Not available in the PDF'.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# 5. Together LLM setup
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # ✅ Serverless model
    temperature=0.3,
    max_tokens=512,
    together_api_key=TOGETHER_API_KEY
)

# 6. QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# 7. Chat loop
while True:
    question = input("Ask a question about the document (type 'exit' to quit):\n>> ")
    if question.lower() == 'exit':
        break
    try:
        answer = qa_chain.invoke(question)
        print("✅ Answer:", answer)
    except Exception as e:
        print("❌ Error:", e)
