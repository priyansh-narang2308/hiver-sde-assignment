import pandas as pd
import re
import os


def clean_text(text):
    """Clean tweet text by removing URLs and extra spaces."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(raw_path="data/raw/twcs.csv", output_path="data/processed/spotify_threads.csv", brand_name="SpotifyCares"):
    """
    Load raw Kaggle dataset, filter for a specific brand, reconstruct threads, and clean text.
    """
    print(f"Loading raw data from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(
            f"Error: Could not find {raw_path}. Is the Kaggle download complete?")
        return

    print(f"Total raw tweets: {len(df)}")

    brand_tweets = df[df['author_id'] == brand_name]
    print(f"Total tweets by {brand_name}: {len(brand_tweets)}")

    customer_tweets_ids = brand_tweets['in_response_to_tweet_id'].dropna().astype(
        int).astype(str).tolist()

    # Create a mapping dictionary for fast lookup
    df['tweet_id_str'] = df['tweet_id'].astype(str)
    customer_tweets = df[df['tweet_id_str'].isin(customer_tweets_ids)].copy()

    # Create a dictionary for quick O(1) lookup: customer_tweet_id -> customer_text
    customer_text_map = dict(
        zip(customer_tweets['tweet_id_str'], customer_tweets['text']))

    # 3. Build the paired dataset
    processed_data = []

    for _, agent_row in brand_tweets.iterrows():
        if pd.isna(agent_row['in_response_to_tweet_id']):
            continue
        customer_id = str(int(agent_row['in_response_to_tweet_id']))
        # If we have the customer's initial tweet in our map
        if customer_id in customer_text_map:
            customer_text = customer_text_map[customer_id]
            agent_text = agent_row['text']

            processed_data.append({
                'customer_tweet_id': customer_id,
                'agent_tweet_id': agent_row['tweet_id'],
                'customer_text': clean_text(customer_text),
                'agent_text': clean_text(agent_text)
            })

    processed_df = pd.DataFrame(processed_data)
    print(
        f"Successfully reconstructed {len(processed_df)} conversation threads.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    preprocess_data()
