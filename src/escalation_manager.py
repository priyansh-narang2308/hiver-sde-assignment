import ollama


class EscalationManager:
    def __init__(self, model_name="gemma4:e2b"):
        self.model_name = model_name
        self.system_prompt = """You are an expert customer support triager for @SpotifyCares.
Your ONLY job is to decide whether a customer tweet needs to be escalated to a human agent, and provide a short reason.

Escalate to a human (Yes) IF:
- The customer is highly frustrated, angry, swearing, or threatening to cancel.
- The issue involves a complex billing dispute or legal threat.
- The customer mentions they have already tried support multiple times without success.

Do NOT escalate (No) IF:
- It is a standard question, basic troubleshooting, or a feature request.
- The tone is neutral or positive.

Output format must be exactly TWO lines:
Line 1: Escalate: [Yes or No]
Line 2: Reason: [One short sentence explaining why]
"""

    def evaluate(self, tweet: str, intent: str) -> dict:
        """
        Evaluates if the tweet should be escalated.
        Returns a dictionary with 'escalate' (bool) and 'reason' (str).
        """
        prompt = f"Intent: {intent}\nCustomer Tweet: \"{tweet}\""

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ])

            output = response['message']['content'].strip().split('\n')

            escalate = False
            reason = ""

            if len(output) > 0 and "yes" in output[0].lower():
                escalate = True

            if len(output) > 1:
                reason = output[1]
                reason = reason.replace("Line 2:", "").replace(
                    "Reason:", "").strip()

            if intent == 'Subscription/Billing' and ('fraud' in tweet.lower() or 'stolen' in tweet.lower()):
                escalate = True
                reason = "Potential fraud or stolen credit card mentioned."

            return {
                "escalate": escalate,
                "reason": reason
            }

        except Exception as e:
            print(f"Error during escalation evaluation: {e}")
            return {"escalate": True, "reason": "System error, safe-fallback to human."}


if __name__ == "__main__":
    manager = EscalationManager()

    test_cases = [
        ("How do I change my playlist cover?", "General Inquiry/Other"),
        ("I've been charged 4 times for premium! I am cancelling right now if this isn't fixed!!",
         "Subscription/Billing"),
        ("When is the new Taylor Swift album dropping on Spotify?",
         "General Inquiry/Other")
    ]

    print("Testing Escalation Manager...\n")
    for tweet, intent in test_cases:
        print(f"Tweet: {tweet}")
        result = manager.evaluate(tweet, intent)
        print(f"Escalate: {result['escalate']}")
        print(f"Reason:   {result['reason']}\n")
        print("-" * 40)
