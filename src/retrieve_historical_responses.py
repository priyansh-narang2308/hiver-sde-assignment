import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# CRITICAL: These must be set BEFORE importing sentence_transformers
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sentence_transformers import SentenceTransformer
class RAGRetriever:
    def __init__(self,
                 metadata_csv="data/processed/rag_metadata.csv",
                 embeddings_npy="data/processed/rag_embeddings.npy"):

        print("Loading RAG Database...")
        self.df = pd.read_csv(metadata_csv)
        self.embeddings = np.load(embeddings_npy)

        print("Initializing Embedding Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Fitting NearestNeighbors Index...")
        self.nn = NearestNeighbors(
            n_neighbors=5, metric='cosine', algorithm='brute')
        self.nn.fit(self.embeddings)
        print("Retriever ready!")

    def retrieve(self, query, top_k=3):
        """
        Embeds the customer query and returns the historical agent responses to the most similar past queries.
        """
        query_vector = self.model.encode([query])

        distances, indices = self.nn.kneighbors(
            query_vector, n_neighbors=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.df.iloc[idx]
            results.append({
                'distance': dist,
                'historical_customer_tweet': row['customer_text'],
                'historical_agent_reply': row['agent_text']
            })

        return results


if __name__ == "__main__":
    retriever = RAGRetriever()

    test_query = "I cannot log into my spotify account, it keeps saying invalid password!"
    print(f"\n--- Testing Query ---")
    print(f"Query: {test_query}\n")

    results = retriever.retrieve(test_query, top_k=2)
    for i, res in enumerate(results):
        print(f"Result {i+1} (Distance: {res['distance']:.4f})")
        print(f"Historic Customer: {res['historical_customer_tweet']}")
        print(f"Historic Agent:    {res['historical_agent_reply']}")
        print("-" * 50)
