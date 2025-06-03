from document_processor import DocumentProcessor
from embeddings import EmbeddingManager
from retriever import Retriever

# Process documents (using previous example data)
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
chunks = processor.process_documents("data")

# Create embedding manager and vector store
embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
embedding_manager.create_vectorstore(chunks)
vectorstore = embedding_manager.get_vectorstore()

# Initialize retriever
retriever = Retriever(vectorstore, top_k=2)

# Test simple retrieval
query = "what is the faculty of Alaa?"
documents = retriever.retrieve(query)

print(f"Query: {query}")
print(f"Retrieved {len(documents)} documents:")
for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc.page_content[:150]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")

# Test retrieval with scores
documents_with_scores = retriever.retrieve_with_scores(query)
print("\nDocuments with similarity scores:")
for i, (doc, score) in enumerate(documents_with_scores):
    print(f"Document {i+1} - Score: {score:.4f}")

# Test MMR retrieval for diversity
mmr_documents = retriever.retrieve_with_mmr(query, diversity=0.7)
print("\nMMR retrieval results:")
for i, doc in enumerate(mmr_documents):
    print(f"Document {i+1}: {doc.page_content[:50]}...")
