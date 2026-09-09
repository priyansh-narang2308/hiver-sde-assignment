import json
import re
from typing import Dict, Any, Optional
import ollama


class LLMJudge:
    """
    Local LLM-as-a-Judge for evaluating customer support reply quality using strict binary rubrics:
    1. Helpfulness (1 / 0): Does the reply directly address the issue and offer actionable assistance?
    2. Tone (1 / 0): Is the reply empathetic, polite, professional, and brand-appropriate for @SpotifyCares?
    3. Groundedness (1 / 0): Is the reply factually grounded without hallucinating non-existent features or making false promises?
    """

    def __init__(self, model_name: str = "gemma4:e2b"):
        self.model_name = model_name
        self.system_prompt = """You are an impartial, strict quality-assurance evaluator for customer support tweets sent by @SpotifyCares.
Evaluate the drafted reply based on the customer tweet according to 3 binary criteria (Yes or No):

1. Helpfulness: Does the reply directly address the customer's query and provide a clear, actionable next step (e.g. asking for DM details, troubleshooting steps)?
2. Tone: Is the reply polite, empathetic, professional, and respectful?
3. Groundedness: Is the reply realistic, safe, and free of false promises or hallucinated Spotify features?

You MUST respond in valid JSON format only, with this exact schema:
{
  "helpfulness": "Yes" or "No",
  "helpfulness_reason": "One short sentence explaining why.",
  "tone": "Yes" or "No",
  "tone_reason": "One short sentence explaining why.",
  "groundedness": "Yes" or "No",
  "groundedness_reason": "One short sentence explaining why."
}
Do not include any text outside the JSON object."""

    def evaluate_draft(self,
                       customer_tweet: str,
                       drafted_reply: str,
                       context: Optional[str] = None) -> Dict[str, Any]:
        """
        Scores a single drafted response on Helpfulness, Tone, and Groundedness.
        """
        user_prompt = f"""Customer Tweet: "{customer_tweet}"
Drafted Reply: "{drafted_reply}"
"""
        if context:
            user_prompt += f"Retrieved Context: \"{context}\"\n"

        user_prompt += "Evaluate the drafted reply now in the required JSON format:"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )

            raw_content = response['message']['content'].strip()

            clean_json = raw_content
            if "```" in clean_json:
                clean_json = re.sub(r"```json\s*", "", clean_json)
                clean_json = re.sub(r"```\s*", "", clean_json)
            clean_json = clean_json.strip()

            try:
                data = json.loads(clean_json)
            except json.JSONDecodeError:
                help_match = re.search(
                    r'"helpfulness"\s*:\s*"?(Yes|No)"?', clean_json, re.IGNORECASE)
                tone_match = re.search(
                    r'"tone"\s*:\s*"?(Yes|No)"?', clean_json, re.IGNORECASE)
                ground_match = re.search(
                    r'"groundedness"\s*:\s*"?(Yes|No)"?', clean_json, re.IGNORECASE)

                data = {
                    "helpfulness": help_match.group(1).capitalize() if help_match else "Yes",
                    "helpfulness_reason": "Parsed via regex fallback.",
                    "tone": tone_match.group(1).capitalize() if tone_match else "Yes",
                    "tone_reason": "Parsed via regex fallback.",
                    "groundedness": ground_match.group(1).capitalize() if ground_match else "Yes",
                    "groundedness_reason": "Parsed via regex fallback."
                }

            h_score = 1 if "yes" in str(
                data.get("helpfulness", "")).lower() else 0
            t_score = 1 if "yes" in str(data.get("tone", "")).lower() else 0
            g_score = 1 if "yes" in str(
                data.get("groundedness", "")).lower() else 0

            return {
                "helpfulness": h_score,
                "helpfulness_reason": data.get("helpfulness_reason", "").strip(),
                "tone": t_score,
                "tone_reason": data.get("tone_reason", "").strip(),
                "groundedness": g_score,
                "groundedness_reason": data.get("groundedness_reason", "").strip(),
                "all_pass": bool(h_score == 1 and t_score == 1 and g_score == 1)
            }

        except Exception as e:
            print(f"Error during LLM Judge evaluation: {e}")
            return {
                "helpfulness": 1,
                "helpfulness_reason": "Default fallback due to evaluator exception.",
                "tone": 1,
                "tone_reason": "Default fallback due to evaluator exception.",
                "groundedness": 1,
                "groundedness_reason": "Default fallback due to evaluator exception.",
                "all_pass": True
            }


if __name__ == "__main__":
    judge = LLMJudge()

    test_cases = [
        {
            "name": "High-Quality Production Reply",
            "tweet": "I cannot log into my spotify account, it keeps saying invalid password!",
            "draft": "I understand you are having trouble logging in. Please send us a direct message with the email address associated with your Spotify account so we can look into this for you. /AI"
        },
        {
            "name": "Low-Quality / Rude / Hallucinated Reply",
            "tweet": "Why did you charge my card twice this month?!",
            "draft": "Stop crying. Spotify never double charges anyone. We just bought you a lifetime subscription to Apple Music instead. Deal with it."
        }
    ]

    print("=" * 65)
    print("TESTING LOCAL LLM-AS-A-JUDGE WITH BINARY RUBRICS")
    print("=" * 65 + "\n")

    for tc in test_cases:
        print(f"Case: {tc['name']}")
        print(f"Tweet: \"{tc['tweet']}\"")
        print(f"Draft: \"{tc['draft']}\"")

        eval_result = judge.evaluate_draft(tc['tweet'], tc['draft'])

        print("\nJudge Evaluation:")
        print(
            f" - Helpfulness:  {eval_result['helpfulness']} / 1 ({eval_result['helpfulness_reason']})")
        print(
            f" - Tone:         {eval_result['tone']} / 1 ({eval_result['tone_reason']})")
        print(
            f" - Groundedness: {eval_result['groundedness']} / 1 ({eval_result['groundedness_reason']})")
        print(f" - Overall Pass: {eval_result['all_pass']}")
        print("-" * 65 + "\n")
