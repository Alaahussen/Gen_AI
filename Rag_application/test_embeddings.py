from document_processor import DocumentProcessor
from embeddings import EmbeddingManager
import os

# Process documents
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
chunks = processor.process_documents("data")

# Create a persist directory
os.makedirs("vectorstore", exist_ok=True)

# Initialize embedding manager with local embeddings
embedding_manager = EmbeddingManager(
    model_name="all-MiniLM-L6-v2",  # Lightweight but effective embedding model
    persist_directory="vectorstore"
)

# Create vector store from documents
embedding_manager.create_vectorstore(chunks)

# Test loading the vector store
embedding_manager = EmbeddingManager(
    model_name="all-MiniLM-L6-v2",
    persist_directory="vectorstore"
)
success = embedding_manager.load_vectorstore()
print(f"Vector store loaded successfully: {success}")

# Get the vector store
vectorstore = embedding_manager.get_vectorstore()
print(f"Vector store contains approximately {vectorstore._collection.count()} documents")
