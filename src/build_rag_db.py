import pandas as pd
import numpy as np
import os


def build_rag_index(threads_csv="data/processed/spotify_threads.csv",
                    embeddings_npy="data/processed/customer_embeddings.npy",
                    golden_csv="data/processed/golden_set.csv",
                    rag_metadata_csv="data/processed/rag_metadata.csv",
                    rag_embeddings_npy="data/processed/rag_embeddings.npy"):

    print("Loading datasets and embeddings...")
    df_threads = pd.read_csv(threads_csv)
    embeddings = np.load(embeddings_npy)

    df_golden = pd.read_csv(golden_csv)
    golden_tweet_ids = set(df_golden['customer_tweet_id'].tolist())

    mask = ~df_threads['customer_tweet_id'].isin(golden_tweet_ids)
    df_rag = df_threads[mask].reset_index(drop=True)
    embeddings_rag = embeddings[mask]

    print(f"Total historical threads: {len(df_threads)}")
    print(
        f"Removed {len(df_threads) - len(df_rag)} golden set examples to prevent data leakage.")
    print(
        f"Building RAG Knowledge Base with {len(df_rag)} historical resolutions...")

    np.save(rag_embeddings_npy, embeddings_rag)
    df_rag.to_csv(rag_metadata_csv, index=False)

    print("RAG Database successfully built! Zero external dependencies needed.")
    print(f"RAG Embeddings saved to {rag_embeddings_npy}")
    print(f"RAG Metadata saved to {rag_metadata_csv}")


if __name__ == "__main__":
    build_rag_index()
