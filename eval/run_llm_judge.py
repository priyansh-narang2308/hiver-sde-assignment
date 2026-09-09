import os
import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm

# Ensure proper path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eval.llm_judge import LLMJudge


def run_llm_judge_evaluation(predictions_csv="eval/benchmark_predictions.csv",
                             output_csv="eval/llm_judge_evaluations.csv",
                             summary_json="eval/judge_summary_metrics.json"):
    print("=" * 70)
    print("LLM-AS-A-JUDGE QUALITY EVALUATION HARNESS")
    print("=" * 70)

    if not os.path.exists(predictions_csv):
        print(f"Error: Predictions file not found at {predictions_csv}.")
        print("Please run 'python eval/run_benchmark.py' first to generate model predictions.")
        return

    print(f"Loading predictions from {predictions_csv}...")
    df = pd.read_csv(predictions_csv)
    print(f"Total benchmark cases to judge: {len(df)}\n")

    judge = LLMJudge()

    systems = [
        ("SpotifySupportAgent", "agent_draft"),
        ("Simple Baseline", "simple_draft"),
        ("Trivial Baseline", "trivial_draft")
    ]

    judged_records = []

    print("Evaluating drafted replies using local LLM-as-a-Judge binary rubrics...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging Replies"):
        tweet = str(row['customer_text'])
        rec = {
            "customer_tweet_id": row.get('customer_tweet_id', idx),
            "customer_text": tweet
        }

        for sys_name, col in systems:
            draft = str(row.get(col, ""))
            res = judge.evaluate_draft(customer_tweet=tweet, drafted_reply=draft)

            prefix = col.replace("_draft", "")
            rec[f"{prefix}_draft"] = draft
            rec[f"{prefix}_helpfulness"] = res["helpfulness"]
            rec[f"{prefix}_help_reason"] = res["helpfulness_reason"]
            rec[f"{prefix}_tone"] = res["tone"]
            rec[f"{prefix}_tone_reason"] = res["tone_reason"]
            rec[f"{prefix}_groundedness"] = res["groundedness"]
            rec[f"{prefix}_ground_reason"] = res["groundedness_reason"]
            rec[f"{prefix}_all_pass"] = int(res["all_pass"])

        judged_records.append(rec)

    df_judged = pd.DataFrame(judged_records)
    df_judged.to_csv(output_csv, index=False)
    print(f"\nDetailed judgment evaluations saved to: {output_csv}")

    # Compute Aggregate Pass Rates per system
    summary_metrics = {}
    print("\n" + "=" * 70)
    print("LLM-AS-A-JUDGE AGGREGATE QUALITY SCORES")
    print("=" * 70)
    print(f"{'System':<25} | {'Helpful %':<11} | {'Tone %':<9} | {'Grounded %':<12} | {'All Pass %':<10}")
    print("-" * 70)

    for sys_name, col in systems:
        prefix = col.replace("_draft", "")
        h_pct = df_judged[f"{prefix}_helpfulness"].mean() * 100
        t_pct = df_judged[f"{prefix}_tone"].mean() * 100
        g_pct = df_judged[f"{prefix}_groundedness"].mean() * 100
        pass_pct = df_judged[f"{prefix}_all_pass"].mean() * 100

        summary_metrics[sys_name] = {
            "helpfulness_pass_pct": float(h_pct),
            "tone_pass_pct": float(t_pct),
            "groundedness_pass_pct": float(g_pct),
            "overall_pass_pct": float(pass_pct)
        }

        print(f"{sys_name:<25} | {h_pct:>9.1f}% | {t_pct:>7.1f}% | {g_pct:>10.1f}% | {pass_pct:>8.1f}%")

    print("=" * 70 + "\n")

    with open(summary_json, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Summary quality metrics saved to: {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluation.")
    parser.add_argument("--predictions_csv", type=str, default="eval/benchmark_predictions.csv",
                        help="Path to predictions CSV.")
    args = parser.parse_args()

    run_llm_judge_evaluation(predictions_csv=args.predictions_csv)
