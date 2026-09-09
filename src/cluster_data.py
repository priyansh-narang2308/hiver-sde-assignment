import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def cluster_embeddings(csv_path="data/processed/spotify_threads.csv",
                       embeddings_path="data/processed/customer_embeddings.npy",
                       output_path="data/processed/spotify_threads_clustered.csv",
                       n_clusters=15):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loading embeddings from {embeddings_path}...")
    try:
        embeddings = np.load(embeddings_path)
    except FileNotFoundError:
        print(
            f"Error: Could not find {embeddings_path}. Did generate_embeddings.py finish?")
        return

    if len(df) != len(embeddings):
        print(
            f"Error: Mismatch in data lengths! Dataframe: {len(df)}, Embeddings: {len(embeddings)}")
        return

    print(
        f"Performing K-Means clustering with {n_clusters} clusters. This uses local CPU (100% free)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    df['cluster_id'] = cluster_labels

    print(f"Saving clustered data to {output_path}...")
    df.to_csv(output_path, index=False)

    print("\nCluster sizes:")
    print(df['cluster_id'].value_counts().sort_index())
    print("\nDone!")


if __name__ == "__main__":
    cluster_embeddings()
