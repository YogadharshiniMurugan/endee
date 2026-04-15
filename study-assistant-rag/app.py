from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample dataset (replace with Endee dataset)
documents = [
    "Artificial Intelligence is the simulation of human intelligence.",
    "Machine Learning is a subset of AI.",
    "Neural networks are used in deep learning.",
]

# Convert to vectors
doc_embeddings = model.encode(documents)

def search(query):
    query_embedding = model.encode([query])
    
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    best_match = documents[np.argmax(similarities)]
    
    return best_match

# Simple CLI
while True:
    query = input("Ask something: ")
    print("Answer:", search(query))
