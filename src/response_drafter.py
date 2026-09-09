import ollama


class ResponseDrafter:
    def __init__(self, model_name="gemma4:e2b"):
        self.model_name = model_name
        self.system_prompt = """You are an expert customer support agent for @SpotifyCares.
Your job is to draft a polite, helpful reply to a customer tweet.

You will be provided with:
1. The Customer Tweet
2. The Intent Category
3. Historical Examples (how agents previously responded to similar issues)

Rules for your drafted reply:
- Tone: Empathetic, helpful, and concise.
- Keep it under 280 characters (it is a tweet).
- Do NOT include hashtags or emojis.
- End with the sign-off "/AI".
- Do NOT output any reasoning, notes, or prefixes like "Reply:" or "Draft:". Just output the exact text of the reply.
- Base your instructions closely on how the historical agents responded.
"""

    def draft(self, customer_tweet: str, intent: str, historical_contexts: list) -> str:
        """
        Drafts a response using the customer tweet and retrieved historical contexts.
        """
        history_str = ""
        for i, ctx in enumerate(historical_contexts):
            history_str += f"Example {i+1}:\nPast Customer: {ctx['historical_customer_tweet']}\nPast Agent: {ctx['historical_agent_reply']}\n\n"

        prompt = f"""Intent: {intent}
Historical Examples of how to handle this:
{history_str}

New Customer Tweet: "{customer_tweet}"

Draft the reply now:"""

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ])

            reply = response['message']['content'].strip()

            prefixes_to_remove = ["Draft:", "Reply:", "Response:", "Agent:"]
            for prefix in prefixes_to_remove:
                if reply.lower().startswith(prefix.lower()):
                    reply = reply[len(prefix):].strip()

            if not reply.endswith("/AI"):
                reply += " /AI"

            return reply

        except Exception as e:
            print(f"Error during drafting: {e}")
            return "Hey there! We are currently looking into this issue. Please bear with us. /AI"


if __name__ == "__main__":
    drafter = ResponseDrafter()

    test_tweet = "I cannot log into my spotify account, it keeps saying invalid password!"
    test_intent = "Login/Account Issue"
    test_contexts = [
        {
            'historical_customer_tweet': "I CAN'T LOG INTO MY SPOTIFY ACCOUNT HELP",
            'historical_agent_reply': "@717691 Hey Marsha! That doesn't sound good. Can you send us a DM with your account's email address? Are you getting any specific error messages while logging in to your account? We'll take a look under the hood /CE"
        }
    ]

    print("Testing Response Drafter...\n")
    print(f"Customer Tweet: {test_tweet}")
    drafted_reply = drafter.draft(test_tweet, test_intent, test_contexts)
    print("\n--- Drafted Reply ---")
    print(drafted_reply)
    print("---------------------\n")
