import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def hr_at_k(ranked_list, relevant_items, k=10):
    hits = 0
    for top_k in ranked_list[:k]:
        if top_k in relevant_items:
            hits += 1
    return 1.0 if hits > 0 else 0.0


def ndcg_at_k(ranked_list, relevant_items, k=10):
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)
    ideal_count = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_list, relevant_items, k=10):
    hits = sum(1 for item in ranked_list[:k] if item in relevant_items)
    return hits / len(relevant_items) if relevant_items else 0.0


def precision_at_k(ranked_list, relevant_items, k=10):
    hits = sum(1 for item in ranked_list[:k] if item in relevant_items)
    return hits / k if k > 0 else 0.0


def evaluate_recommendations(y_true, y_pred, y_scores, k_values=(5, 10, 20)):
    metrics = {}

    metrics["auc"] = roc_auc_score(y_true, y_scores)
    metrics["accuracy"] = accuracy_score(y_true, (y_scores >= 0.5).astype(int))
    metrics["precision"] = precision_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
    metrics["recall"] = recall_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
    metrics["f1"] = f1_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)

    true_positives = np.where(y_true > 0.5)[0]
    ranked = np.argsort(y_scores)[::-1]

    for k in k_values:
        metrics[f"hr@{k}"] = hr_at_k(ranked, true_positives, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, true_positives, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked, true_positives, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked, true_positives, k)

    return metrics


def evaluate_model_on_test(model, test_loader, device):
    import torch
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            cat_inputs, num_input, labels = [b.to(device) for b in batch]
            outputs = model(cat_inputs, num_input)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return evaluate_recommendations(np.array(all_labels), (np.array(all_preds) >= 0.5).astype(int), np.array(all_preds))


def print_evaluation_report(metrics):
    print("\n" + "=" * 50)
    print("Model Evaluation Report")
    print("=" * 50)
    print(f"AUC-ROC:     {metrics.get('auc', 0):.4f}")
    print(f"Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"Precision:   {metrics.get('precision', 0):.4f}")
    print(f"Recall:      {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:    {metrics.get('f1', 0):.4f}")
    print("-" * 30)
    for k in [5, 10, 20]:
        print(f"HR@{k}:       {metrics.get(f'hr@{k}', 0):.4f}")
        print(f"NDCG@{k}:     {metrics.get(f'ndcg@{k}', 0):.4f}")
        print(f"Recall@{k}:   {metrics.get(f'recall@{k}', 0):.4f}")
        print(f"Precision@{k}:{metrics.get(f'precision@{k}', 0):.4f}")
    print("=" * 50)
