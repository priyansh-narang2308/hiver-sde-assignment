import pandas as pd
import os

def sample_golden_set(input_path="data/processed/spotify_threads_clustered.csv", 
                      output_path="data/processed/golden_set_unlabeled.csv", 
                      sample_size=200):
    """
    Perform stratified sampling based on the K-Means cluster IDs to extract a 
    diverse, highly representative 'golden set' of 200 examples.
    """
    print(f"Loading clustered data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Ensure Task 10 completed successfully.")
        return

    # Drop any rows where customer text is empty to ensure high-quality labeling
    df = df.dropna(subset=['customer_text'])
    
    # Calculate how many samples we need from each cluster
    total_population = len(df)
    cluster_counts = df['cluster_id'].value_counts()
    
    sampled_dfs = []
    current_sample_count = 0
    
    for cluster_id, count in cluster_counts.items():
        # Proportional allocation
        proportion = count / total_population
        n_samples_for_cluster = int(round(proportion * sample_size))
        
        # Ensure we don't oversample if rounding pushes us slightly over
        if current_sample_count + n_samples_for_cluster > sample_size:
            n_samples_for_cluster = sample_size - current_sample_count
            
        if n_samples_for_cluster > 0:
            cluster_df = df[df['cluster_id'] == cluster_id]
            # Randomly sample without replacement
            sampled = cluster_df.sample(n=n_samples_for_cluster, random_state=42)
            sampled_dfs.append(sampled)
            current_sample_count += n_samples_for_cluster
            
    golden_df = pd.concat(sampled_dfs).reset_index(drop=True)
    
    # Shuffle the final dataset so clusters aren't grouped together during manual labeling
    golden_df = golden_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # We only need the tweet IDs, texts, and empty columns for the human labeler
    golden_df = golden_df[['customer_tweet_id', 'agent_tweet_id', 'customer_text', 'agent_text', 'cluster_id']]
    golden_df['intent_label'] = ""
    golden_df['escalate_label'] = ""
    golden_df['escalation_reason'] = ""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    golden_df.to_csv(output_path, index=False)
    
    print(f"Successfully sampled {len(golden_df)} diverse examples.")
    print(f"Saved unlabeled golden set to {output_path}")

if __name__ == "__main__":
    sample_golden_set()
