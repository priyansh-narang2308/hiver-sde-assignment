import os
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


class SimpleBaselineAgent:
    """
    Simple Baseline Agent:
    - Intent Classifier: TF-IDF (3000 n-grams) + Logistic Regression.
    - Escalation Router: TF-IDF + Logistic Regression with balanced class weights.
    - Drafter: TF-IDF Nearest-Neighbor Retrieval over historical resolutions (Exact Verbatim past agent reply, no LLM synthesis).
    """

    def __init__(self,
                 models_dir: str = "data/models",
                 rag_metadata_csv: str = "data/processed/rag_metadata.csv"):
        print("Loading Simple Baseline Models...")
        vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        intent_path = os.path.join(models_dir, "simple_intent_clf.joblib")
        esc_path = os.path.join(models_dir, "simple_escalate_clf.joblib")

        self.vectorizer = joblib.load(vec_path)
        self.clf_intent = joblib.load(intent_path)
        self.clf_escalate = joblib.load(esc_path)

        print("Indexing historical replies for verbatim retrieval drafting...")
        self.df_rag = pd.read_csv(rag_metadata_csv)
        self.historical_tfidf = self.vectorizer.transform(
            self.df_rag['customer_text'].astype(str).values
        )
        print("Simple Baseline Agent Ready!\n")

    def process(self, customer_tweet: str, top_k: int = 1) -> Dict[str, Any]:
        """
        Executes the classical ML simple baseline pipeline on an incoming customer tweet.
        Matches the exact output contract of SpotifySupportAgent.
        """
        tweet_tfidf = self.vectorizer.transform([customer_tweet])

        # 1. Intent Classification
        predicted_intent = self.clf_intent.predict(tweet_tfidf)[0]

        # 2. Escalation Routing
        esc_pred = self.clf_escalate.predict(tweet_tfidf)[0]
        escalate = bool(esc_pred == 1)

        reason = (
            f"Simple baseline classifier detected escalation triggers for intent: {predicted_intent}."
            if escalate
            else "Standard query, no escalation triggers detected by baseline classifier."
        )

        # 3. Verbatim Historical Retrieval (No LLM generation)
        sim_scores = cosine_similarity(
            tweet_tfidf, self.historical_tfidf).flatten()
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        retrieved_contexts = []
        verbatim_draft = ""

        if len(top_indices) > 0:
            best_idx = top_indices[0]
            matched_row = self.df_rag.iloc[best_idx]
            verbatim_draft = str(matched_row['agent_text'])
            retrieved_contexts.append({
                'distance': float(1.0 - sim_scores[best_idx]),
                'historical_customer_tweet': str(matched_row['customer_text']),
                'historical_agent_reply': verbatim_draft
            })
        else:
            verbatim_draft = "Hey there! Please send us a DM with your account details so we can help. /AI"

        return {
            "customer_tweet": customer_tweet,
            "predicted_intent": predicted_intent,
            "escalate": escalate,
            "escalation_reason": reason,
            "retrieved_context": retrieved_contexts,
            "drafted_reply": verbatim_draft
        }


if __name__ == "__main__":
    baseline = SimpleBaselineAgent()

    test_queries = [
        "I cannot log into my spotify account, it keeps saying invalid password!",
        "Why was I charged 3 times for premium this month? Refund me immediately or I am canceling!",
        "How do I create a collaborative playlist with my friend?"
    ]

    print("=" * 60)
    print("TESTING SIMPLE BASELINE AGENT (TF-IDF + LOGISTIC REGRESSION + VERBATIM RETRIEVAL)")
    print("=" * 60 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}] Incoming Tweet: {query}")
        result = baseline.process(query)
        print(f" -> Predicted Intent:   {result['predicted_intent']}")
        print(f" -> Escalate to Human:  {result['escalate']}")
        print(f" -> Escalation Reason:  {result['escalation_reason']}")
        print(f" -> Verbatim Match:\n    \"{result['drafted_reply']}\"")
        print("-" * 60 + "\n")
