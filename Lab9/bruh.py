import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()
# 1. SETUP: Replace with your actual HF token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("lab9_key")


# 2. CREATE DATA: If knowledge.txt doesn't exist, this creates it
with open("knowledge.txt", "w") as f:
    f.write("The 2026 Space Olympics are being held on the Moon in the Shackleton Crater.\n")
    f.write("The official mascot is a robotic rabbit named 'Luna-Hop'.\n")
    f.write("The main event is the low-gravity high jump competition.")

# 3. LOAD & CHUNK: Prepare the data
loader = TextLoader("knowledge.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)

# 4. EMBEDDINGS: Turn text into math (Vectors)
# This downloads a small model (~80MB) to your Mac locally
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)

# 5. LLM SETUP: Using Llama 3.1 (Most reliable for free tier in 2026)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)
chat_model = ChatHuggingFace(llm=llm)

# 6. THE RAG CHAIN: Connect everything
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

# 7. RUN THE QUERY
query = "What is the mascot of the 2026 Space Olympics and where is it held?"
response = qa_chain.invoke(query)

print("\n" + "="*30)
print("RAG SYSTEM OUTPUT:")
print("="*30)
print(response['result'])

