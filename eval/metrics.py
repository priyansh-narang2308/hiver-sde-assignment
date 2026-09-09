from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def normalize_boolean_labels(labels: Union[List[Any], pd.Series, np.ndarray]) -> List[bool]:
    """
    Normalizes diverse boolean label formats ('Yes', 'No', 1, 0, True, False) into standard booleans.
    """
    normalized = []
    for val in labels:
        if isinstance(val, bool):
            normalized.append(val)
        elif isinstance(val, (int, float)):
            normalized.append(bool(val == 1))
        elif isinstance(val, str):
            normalized.append(val.strip().lower() in ['yes', 'true', '1'])
        else:
            normalized.append(False)
    return normalized


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes classification metrics for Intent prediction:
    - Overall Accuracy
    - Macro Precision, Recall, and F1-Score
    - Per-class precision, recall, and F1 breakdown
    """
    y_true_str = [str(y).strip() for y in y_true]
    y_pred_str = [str(y).strip() for y in y_pred]

    acc = accuracy_score(y_true_str, y_pred_str)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_str, y_pred_str, average='macro', zero_division=0
    )

    class_report = classification_report(
        y_true_str, y_pred_str, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class_report": class_report
    }


def compute_escalation_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, Any]:
    """
    Computes evaluation metrics for Human Escalation triage:
    - Accuracy
    - Escalation Recall (Primary safety metric: % of critical/angry tickets successfully caught)
    - Escalation Precision
    - Escalation F1-score
    - False Negative Rate (FNR = 1 - Recall)
    - False Positive Rate (FPR)
    """
    y_true_bool = normalize_boolean_labels(y_true)
    y_pred_bool = normalize_boolean_labels(y_pred)

    acc = accuracy_score(y_true_bool, y_pred_bool)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bool, y_pred_bool, average='binary', pos_label=True, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true_bool, y_pred_bool, labels=[False, True]
    ).ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "escalation_recall": float(r),
        "escalation_precision": float(p),
        "escalation_f1": float(f1),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def evaluate_system(y_true_intent: List[str],
                    y_pred_intent: List[str],
                    y_true_escalate: List[Any],
                    y_pred_escalate: List[Any]) -> Dict[str, Any]:
    """
    Comprehensive evaluation of intent and escalation decisions against ground truth.
    """
    intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent)
    escalation_metrics = compute_escalation_metrics(
        y_true_escalate, y_pred_escalate)

    return {
        "intent": intent_metrics,
        "escalation": escalation_metrics
    }


def print_evaluation_report(metrics: Dict[str, Any], system_name: str = "System"):
    """
    Prints a formatted executive evaluation summary.
    """
    intent = metrics["intent"]
    esc = metrics["escalation"]

    print("\n" + "=" * 65)
    print(f" EVALUATION REPORT: {system_name.upper()}")
    print("=" * 65)

    print("\n[1] INTENT CLASSIFICATION METRICS")
    print(f" - Overall Accuracy:        {intent['accuracy'] * 100:.2f}%")
    print(f" - Macro-F1 Score:          {intent['macro_f1'] * 100:.2f}%")
    print(
        f" - Macro Precision:         {intent['macro_precision'] * 100:.2f}%")
    print(f" - Macro Recall:            {intent['macro_recall'] * 100:.2f}%")

    print("\n[2] HUMAN ESCALATION TRIAGE METRICS")
    print(
        f" - Escalation Recall:       {esc['escalation_recall'] * 100:.2f}% (CRITICAL: % angry/billing caught)")
    print(
        f" - Escalation Precision:    {esc['escalation_precision'] * 100:.2f}%")
    print(f" - Escalation F1-Score:     {esc['escalation_f1'] * 100:.2f}%")
    print(
        f" - False Negative Rate:     {esc['false_negative_rate'] * 100:.2f}% (Uncaught escalations)")
    print(
        f" - False Positive Rate:     {esc['false_positive_rate'] * 100:.2f}%")
    print(
        f" - Confusion Matrix:        TP={esc['true_positives']}, FP={esc['false_positives']}, TN={esc['true_negatives']}, FN={esc['false_negatives']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print("Testing Automated Evaluation Metrics module...")

    mock_true_intent = [
        'Login/Account Issue', 'Subscription/Billing', 'Audio/Playback Issue',
        'Feature Request', 'General Inquiry/Other', 'Subscription/Billing'
    ]
    mock_pred_intent = [
        'Login/Account Issue', 'Subscription/Billing', 'Audio/Playback Issue',
        'General Inquiry/Other', 'General Inquiry/Other', 'Subscription/Billing'
    ]

    mock_true_escalate = ['No', 'Yes', 'No', 'No', 'No', 'Yes']
    mock_pred_escalate = ['No', 'Yes', 'No', 'No', 'No', 'No']

    metrics = evaluate_system(
        mock_true_intent, mock_pred_intent,
        mock_true_escalate, mock_pred_escalate
    )

    print_evaluation_report(metrics, system_name="Mock Benchmark Model")
