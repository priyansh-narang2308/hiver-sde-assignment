import ollama

class IntentClassifier:
    def __init__(self, model_name="gemma4:e2b"):
        self.model_name = model_name
        self.valid_intents = [
            'Login/Account Issue',
            'Audio/Playback Issue',
            'Subscription/Billing',
            'Feature Request',
            'General Inquiry/Other'
        ]

        self.system_prompt = f"""You are an expert customer support routing AI for @SpotifyCares.
Your ONLY job is to classify the customer's tweet into exactly one of the following categories:
{', '.join(self.valid_intents)}

Rules:
1. Output ONLY the category name. Do not output any other text, punctuation, or explanation.
2. If the tweet mentions passwords, logging in, or account access, classify as 'Login/Account Issue'.
3. If the tweet mentions skipping, stuttering, playing songs, or audio quality, classify as 'Audio/Playback Issue'.
4. If the tweet mentions premium, charges, credit cards, or billing, classify as 'Subscription/Billing'.
5. If the tweet asks for a new feature, classify as 'Feature Request'.
6. If it does not clearly fit the above, classify as 'General Inquiry/Other'."""

    def classify(self, tweet: str) -> str:
        """
        Classifies the tweet using the local Ollama model.
        Forces the output to match one of the valid intents.
        """
        prompt = f"Customer Tweet: \"{tweet}\""

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ])

            output = response['message']['content'].strip()

            for valid_intent in self.valid_intents:
                if valid_intent.lower() in output.lower():
                    return valid_intent

            return 'General Inquiry/Other'

        except Exception as e:
            print(f"Error during classification: {e}")
            return 'General Inquiry/Other'


if __name__ == "__main__":
    classifier = IntentClassifier()

    test_tweets = [
        "Why did you charge my card twice this month?!",
        "The app keeps crashing when I try to play my Discover Weekly.",
        "I can't remember my password and the reset email isn't arriving."
    ]

    print("Testing Intent Classifier...\n")
    for tweet in test_tweets:
        print(f"Tweet: {tweet}")
        intent = classifier.classify(tweet)
        print(f"Predicted Intent: {intent}\n")
        print("-" * 40)
