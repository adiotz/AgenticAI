from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. Initialize Free Local Embeddings
# This downloads a small model (~80MB) to your Mac once.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Your Document Corpus
documents = [
    Document(page_content="The Great Wall of China is over 13,000 miles long."),
    Document(page_content="The Eiffel Tower is located in Paris, France."),
    Document(page_content="Python is a popular programming language for AI."),
    Document(page_content="Quantum computing uses qubits for complex math.")
]

# 3. Create the Vector Store (FAISS)
# This will run locally and cost $0.00
vector_store = FAISS.from_documents(documents, embeddings)

# 4. Search Function
def free_semantic_search(query):
    results = vector_store.similarity_search(query, k=1)
    return results[0].page_content

# --- Test ---
print("--- Lab 3: Free Semantic Search ---")
q = '''
what do you mean by qubits
'''
print(f"Query: {q}\nResult: {free_semantic_search(q)}")