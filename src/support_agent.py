import os
import sys
from typing import Dict, Any

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress TensorFlow warnings and configure environment before model imports
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure robust imports whether run from project root or inside src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from src.intent_classifier import IntentClassifier
    from src.escalation_manager import EscalationManager
    from src.retrieve_historical_responses import RAGRetriever
    from src.response_drafter import ResponseDrafter
except ImportError:
    from intent_classifier import IntentClassifier
    from escalation_manager import EscalationManager
    from retrieve_historical_responses import RAGRetriever
    from response_drafter import ResponseDrafter


class SpotifySupportAgent:
    """
    End-to-end AI Customer Support Agent pipeline for @SpotifyCares.
    Orchestrates Intent Classification, Human Escalation Triage, RAG Retrieval, and Response Drafting.
    """
    def __init__(self, model_name: str = "gemma4:e2b"):
        print("Initializing SpotifySupportAgent Pipeline...")
        self.classifier = IntentClassifier(model_name=model_name)
        self.escalation_manager = EscalationManager(model_name=model_name)
        self.retriever = RAGRetriever()
        self.drafter = ResponseDrafter(model_name=model_name)
        print("SpotifySupportAgent Pipeline Ready!\n")

    def process(self, customer_tweet: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Executes the full triage and response generation pipeline for an incoming customer tweet.
        """
        # Step 1: Intent Classification
        intent = self.classifier.classify(customer_tweet)

        # Step 2: Escalation Triage & Evaluation
        escalation_info = self.escalation_manager.evaluate(customer_tweet, intent)
        escalate = escalation_info["escalate"]
        reason = escalation_info["reason"]

        # Step 3: Historical RAG Retrieval
        retrieved_contexts = self.retriever.retrieve(customer_tweet, top_k=top_k)

        # Step 4: Context-Grounded Response Drafting
        drafted_reply = self.drafter.draft(customer_tweet, intent, retrieved_contexts)

        return {
            "customer_tweet": customer_tweet,
            "predicted_intent": intent,
            "escalate": escalate,
            "escalation_reason": reason,
            "retrieved_context": retrieved_contexts,
            "drafted_reply": drafted_reply
        }


if __name__ == "__main__":
    agent = SpotifySupportAgent()

    test_queries = [
        "I cannot log into my spotify account, it keeps saying invalid password!",
        "Why was I charged 3 times for premium this month? Refund me immediately or I am canceling!",
        "How do I create a collaborative playlist with my friend?"
    ]

    print("=" * 60)
    print("RUNNING END-TO-END SPOTIFY SUPPORT AGENT PIPELINE TEST")
    print("=" * 60 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}] Incoming Tweet: {query}")
        result = agent.process(query)

        print(f" -> Predicted Intent:   {result['predicted_intent']}")
        print(f" -> Escalate to Human:  {result['escalate']}")
        print(f" -> Escalation Reason:  {result['escalation_reason']}")
        print(f" -> Top Retrieved Match: (Dist: {result['retrieved_context'][0]['distance']:.4f}) {result['retrieved_context'][0]['historical_agent_reply'][:75]}...")
        print(f" -> Final Drafted Reply:\n    \"{result['drafted_reply']}\"")
        print("-" * 60 + "\n")
