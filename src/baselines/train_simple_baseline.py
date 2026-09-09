import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def assign_weak_labels(text: str):
    """
    Applies high-precision heuristic rules to generate training labels from raw historical customer tweets.
    """
    t = text.lower()

    # Intent labeling rules
    if any(k in t for k in ['password', 'log in', 'login', 'sign in', 'account', 'locked', 'reset password', 'cant access', 'email']):
        intent = 'Login/Account Issue'
    elif any(k in t for k in ['charge', 'charged', 'refund', 'premium', 'bill', 'billing', 'subscription', 'credit card', 'pay', 'payment', 'receipt', 'money', 'cancel subscription']):
        intent = 'Subscription/Billing'
    elif any(k in t for k in ['crash', 'skipping', 'stutter', 'song', 'play', 'pause', 'track', 'album', 'audio', 'sound', 'offline', 'download', 'bluetooth', 'listen']):
        intent = 'Audio/Playback Issue'
    elif any(k in t for k in ['please add', 'feature', 'can you add', 'support for', 'dark mode', 'lyrics', 'wish spotify', 'option to']):
        intent = 'Feature Request'
    else:
        intent = 'General Inquiry/Other'

    # Escalation labeling rules
    frustrated_cues = ['cancel', 'cancelling', 'refund', 'charged twice', 'charged 3 times', 'charged 4 times',
                       'unauthorized', 'stolen', 'fraud', 'scam', 'horrible', 'worst', 'disgusted', 'lawsuit',
                       'sue', 'fix this right now', 'fix this immediately', 'hate']
    escalate = "Yes" if any(k in t for k in frustrated_cues) else "No"

    return intent, escalate


def train_simple_baseline(threads_csv="data/processed/spotify_threads.csv",
                          golden_csv="data/processed/golden_set.csv",
                          models_dir="data/models"):
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data for Simple Baseline training...")
    df = pd.read_csv(threads_csv)

    # Exclude Golden Set to strictly prevent data leakage
    if os.path.exists(golden_csv):
        df_golden = pd.read_csv(golden_csv)
        golden_ids = set(df_golden['customer_tweet_id'].tolist())
        df = df[~df['customer_tweet_id'].isin(golden_ids)].reset_index(drop=True)
        print(f"Excluded {len(golden_ids)} Golden Set rows to preserve strict evaluation integrity.")

    # Sample 5,000 tweets for fast, robust training
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()

    print(f"Generating training labels on {sample_size} threads using domain heuristics...")
    labels = df_sample['customer_text'].astype(str).apply(assign_weak_labels)
    df_sample['intent'] = [l[0] for l in labels]
    df_sample['escalate'] = [l[1] for l in labels]

    X = df_sample['customer_text'].astype(str).values
    y_intent = df_sample['intent'].values
    y_escalate = (df_sample['escalate'] == "Yes").astype(int).values

    print("\nTraining TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)

    # Train-test split for quick validation
    X_train, X_val, y_int_train, y_int_val, y_esc_train, y_esc_val = train_test_split(
        X_tfidf, y_intent, y_escalate, test_size=0.2, random_state=42
    )

    print("Training Logistic Regression Intent Classifier...")
    clf_intent = LogisticRegression(max_iter=1000, C=1.0)
    clf_intent.fit(X_train, y_int_train)
    val_acc_intent = clf_intent.score(X_val, y_int_val)
    print(f" -> Intent Validation Accuracy: {val_acc_intent * 100:.2f}%")

    print("Training Logistic Regression Escalation Router...")
    clf_escalate = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_escalate.fit(X_train, y_esc_train)
    val_acc_escalate = clf_escalate.score(X_val, y_esc_val)
    print(f" -> Escalation Validation Accuracy: {val_acc_escalate * 100:.2f}%")

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(clf_intent, os.path.join(models_dir, "simple_intent_clf.joblib"))
    joblib.dump(clf_escalate, os.path.join(models_dir, "simple_escalate_clf.joblib"))

    print(f"\nAll baseline models saved successfully in {models_dir}/")


if __name__ == "__main__":
    train_simple_baseline()
