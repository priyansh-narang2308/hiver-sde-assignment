import pandas as pd
import ollama
from tqdm import tqdm
import os


def auto_label_golden_set(input_path="data/processed/golden_set_unlabeled.csv",
                          output_path="data/processed/golden_set.csv",
                          model_name="gemma4:e2b"):

    print(f"Loading unlabelled dataset from {input_path}...")
    df = pd.read_csv(input_path)

    if 'intent_label' not in df.columns:
        df['intent_label'] = ""
        df['escalate_label'] = ""
        df['escalation_reason'] = ""

    df['intent_label'] = df['intent_label'].astype(str)
    df['escalate_label'] = df['escalate_label'].astype(str)
    df['escalation_reason'] = df['escalation_reason'].astype(str)

    unlabeled_idx = df[df['intent_label'] == ""].index

    if len(unlabeled_idx) == 0:
        print("All examples are already labeled!")
        return

    print(
        f"Auto-labeling {len(unlabeled_idx)} examples using local model {model_name}...")
    print("This simulates the 'hand-labeled' process. You can manually review the CSV afterwards.")

    system_prompt = """You are an expert customer support analyst for @SpotifyCares.
Read the customer tweet and classify it.
Output exactly THREE lines. Do not add any conversational text.
Line 1: Intent (Must be exactly one of: 'Login/Account Issue', 'Audio/Playback Issue', 'Subscription/Billing', 'Feature Request', 'General Inquiry/Other')
Line 2: Escalate (Must be exactly 'Yes' or 'No'. Escalate if the user is extremely angry, threatening to cancel, or has a complex billing issue)
Line 3: Reason (If Escalate is Yes, 1 short sentence why. If No, leave blank)"""

    for idx in tqdm(unlabeled_idx):
        tweet = df.loc[idx, 'customer_text']

        prompt = f"Customer Tweet:\n\"{tweet}\""

        try:
            response = ollama.chat(model=model_name, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ])

            output = response['message']['content'].strip().split('\n')

            if len(output) >= 2:
                intent = output[0].replace('Intent:', '').strip()
                escalate = output[1].replace('Escalate:', '').strip()
                reason = output[2].replace(
                    'Reason:', '').strip() if len(output) > 2 else ""

                intent = intent.strip("'* ")
                escalate = "Yes" if "yes" in escalate.lower() else "No"

                df.at[idx, 'intent_label'] = intent
                df.at[idx, 'escalate_label'] = escalate
                df.at[idx, 'escalation_reason'] = reason
            else:
                df.at[idx, 'intent_label'] = "General Inquiry/Other"
                df.at[idx, 'escalate_label'] = "No"

        except Exception as e:
            print(f"Error on index {idx}: {e}")

        df.to_csv(output_path, index=False)

    print(f"Successfully auto-labeled and saved to {output_path}!")


if __name__ == "__main__":
    auto_label_golden_set()
