import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os


def generate_embeddings(input_csv="data/processed/spotify_threads.csv", output_npy="data/processed/customer_embeddings.npy"):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    customer_texts = df['customer_text'].fillna("").tolist()

    print("Loading embedding model (all-MiniLM-L6-v2)...")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(
        f"Generating embeddings for {len(customer_texts)} queries (this may take a minute)...")
    embeddings = model.encode(customer_texts, show_progress_bar=True)

    print(f"Saving embeddings to {output_npy}...")
    np.save(output_npy, embeddings)
    print("Done!")


if __name__ == "__main__":
    generate_embeddings()
