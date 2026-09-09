from eval.metrics import evaluate_system, print_evaluation_report
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleBaselineAgent
from src.support_agent import SpotifySupportAgent
import warnings
import os
import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm

os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings('ignore', category=RuntimeWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_benchmark(golden_set_path="data/processed/golden_set.csv",
                  output_csv="eval/benchmark_predictions.csv",
                  output_metrics_json="eval/benchmark_metrics.json",
                  sample_size: int = 20):
    print("=" * 70)
    print("SPOTIFY SUPPORT AGENT VS BASELINES BENCHMARK RUNNER")
    print("=" * 70)

    print(f"Loading Golden Set from {golden_set_path}...")
    df_golden = pd.read_csv(golden_set_path)

    if sample_size and sample_size < len(df_golden):
        print(
            f"Sampling {sample_size} diverse test cases across all intent clusters...")
        sample_df = df_golden.groupby('intent_label', group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(len(x) / len(df_golden) * sample_size)), random_state=42)
        )
        if len(sample_df) < sample_size:
            remaining = df_golden[~df_golden.index.isin(sample_df.index)]
            needed = sample_size - len(sample_df)
            sample_df = pd.concat(
                [sample_df, remaining.sample(n=needed, random_state=42)])
        df_test = sample_df.reset_index(drop=True)
    else:
        df_test = df_golden.copy()

    print(f"Benchmark Test Set Size: {len(df_test)} customer queries.\n")

    # Initialize all 3 systems
    print("Initializing Benchmark Systems...")
    agent = SpotifySupportAgent()
    simple_baseline = SimpleBaselineAgent()
    trivial_baseline = TrivialBaselineAgent()

    results = []

    print(
        f"\nExecuting benchmark across {len(df_test)} test queries (this will take a few minutes)...")
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Evaluating Systems"):
        tweet = str(row['customer_text'])
        gt_intent = str(row['intent_label'])
        gt_esc = str(row['escalate_label'])
        gt_reason = str(row.get('escalation_reason', ''))

        # 1. SpotifySupportAgent
        agent_res = agent.process(tweet, top_k=2)

        # 2. Simple Baseline (TF-IDF + Logistic Regression + Verbatim Match)
        simple_res = simple_baseline.process(tweet, top_k=1)

        # 3. Trivial Baseline (Majority class + Constant Escalate)
        trivial_res = trivial_baseline.process(tweet)

        results.append({
            "customer_tweet_id": row.get('customer_tweet_id', idx),
            "customer_text": tweet,
            "ground_truth_intent": gt_intent,
            "ground_truth_escalate": gt_esc,
            "ground_truth_reason": gt_reason,
            # Agent Predictions
            "agent_intent": agent_res["predicted_intent"],
            "agent_escalate": "Yes" if agent_res["escalate"] else "No",
            "agent_reason": agent_res["escalation_reason"],
            "agent_draft": agent_res["drafted_reply"],
            # Simple Baseline Predictions
            "simple_intent": simple_res["predicted_intent"],
            "simple_escalate": "Yes" if simple_res["escalate"] else "No",
            "simple_reason": simple_res["escalation_reason"],
            "simple_draft": simple_res["drafted_reply"],
            # Trivial Baseline Predictions
            "trivial_intent": trivial_res["predicted_intent"],
            "trivial_escalate": "Yes" if trivial_res["escalate"] else "No",
            "trivial_reason": trivial_res["escalation_reason"],
            "trivial_draft": trivial_res["drafted_reply"]
        })

    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nAll predictions successfully saved to: {output_csv}")

    # Compute Comparative Metrics
    y_true_intent = df_results["ground_truth_intent"].tolist()
    y_true_esc = df_results["ground_truth_escalate"].tolist()

    agent_metrics = evaluate_system(
        y_true_intent, df_results["agent_intent"].tolist(),
        y_true_esc, df_results["agent_escalate"].tolist()
    )

    simple_metrics = evaluate_system(
        y_true_intent, df_results["simple_intent"].tolist(),
        y_true_esc, df_results["simple_escalate"].tolist()
    )

    trivial_metrics = evaluate_system(
        y_true_intent, df_results["trivial_intent"].tolist(),
        y_true_esc, df_results["trivial_escalate"].tolist()
    )

    all_metrics = {
        "spotify_support_agent": agent_metrics,
        "simple_baseline": simple_metrics,
        "trivial_baseline": trivial_metrics
    }

    with open(output_metrics_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics JSON saved to: {output_metrics_json}")

    # Print Executive Comparison Reports
    print_evaluation_report(trivial_metrics, system_name="Trivial Baseline")
    print_evaluation_report(
        simple_metrics, system_name="Simple Baseline (TF-IDF + LogReg)")
    print_evaluation_report(
        agent_metrics, system_name="SpotifySupportAgent (LLM + RAG)")

    print("=" * 70)
    print("BENCHMARK COMPARISON SUMMARY TABLE")
    print("=" * 70)
    print(f"{'System':<25} | {'Intent Acc':<11} | {'Intent F1':<10} | {'Esc Recall':<11} | {'Esc F1':<8}")
    print("-" * 70)

    systems = [
        ("Trivial Baseline", trivial_metrics),
        ("Simple Baseline", simple_metrics),
        ("SpotifySupportAgent", agent_metrics)
    ]

    for name, m in systems:
        acc = m["intent"]["accuracy"] * 100
        f1 = m["intent"]["macro_f1"] * 100
        rec = m["escalation"]["escalation_recall"] * 100
        esc_f1 = m["escalation"]["escalation_f1"] * 100
        print(
            f"{name:<25} | {acc:>9.1f}% | {f1:>8.1f}% | {rec:>9.1f}% | {esc_f1:>6.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation over Golden Set.")
    parser.add_argument("--sample_size", type=int, default=20,
                        help="Number of diverse test queries to evaluate (default: 20 for fast comprehensive benchmark). Use 0 for full golden set.")
    args = parser.parse_args()

    sample_size = None if args.sample_size == 0 else args.sample_size
    run_benchmark(sample_size=sample_size)
