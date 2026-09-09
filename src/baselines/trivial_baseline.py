from typing import Dict, Any


class TrivialBaselineAgent:
    """
    Trivial Baseline Agent for benchmark comparisons.
    Follows a zero-intelligence policy:
    1. Intent: Always predicts the majority class ('Audio/Playback Issue').
    2. Escalation: Constant policy (always True, escalating all queries to human).
    3. Response Drafter: Returns a fixed canned response.
    """

    def __init__(self,
                 majority_intent: str = "Audio/Playback Issue",
                 constant_escalate: bool = True):
        self.majority_intent = majority_intent
        self.constant_escalate = constant_escalate
        self.canned_reply = (
            "Hi there! Thanks for reaching out to Spotify Cares. "
            "Please send us a direct message with your account email address so we can take a closer look. /AI"
        )

    def process(self, customer_tweet: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Executes the trivial baseline pipeline on an incoming customer tweet.
        Matches the exact output contract of SpotifySupportAgent.
        """
        reason = (
            "Trivial baseline constant policy (always escalate to human for safety)."
            if self.constant_escalate
            else "Trivial baseline constant policy (never escalate)."
        )

        return {
            "customer_tweet": customer_tweet,
            "predicted_intent": self.majority_intent,
            "escalate": self.constant_escalate,
            "escalation_reason": reason,
            "retrieved_context": [],
            "drafted_reply": self.canned_reply
        }


if __name__ == "__main__":
    baseline = TrivialBaselineAgent()

    test_queries = [
        "I cannot log into my spotify account, it keeps saying invalid password!",
        "Why was I charged 3 times for premium this month? Refund me immediately or I am canceling!",
        "How do I create a collaborative playlist with my friend?"
    ]

    print("=" * 60)
    print("TESTING TRIVIAL BASELINE AGENT")
    print("=" * 60 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}] Incoming Tweet: {query}")
        result = baseline.process(query)
        print(f" -> Predicted Intent:   {result['predicted_intent']}")
        print(f" -> Escalate to Human:  {result['escalate']}")
        print(f" -> Escalation Reason:  {result['escalation_reason']}")
        print(f" -> Final Drafted Reply:\n    \"{result['drafted_reply']}\"")
        print("-" * 60 + "\n")
