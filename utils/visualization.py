import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from config import OUTPUT_DIR

plt.rcParams["axes.unicode_minus"] = False

_FONT_PATH = None
for f in fm.findSystemFonts():
    if "simhei" in f.lower() or "microsoft yahei" in f.lower() or "simsun" in f.lower():
        try:
            fm.fontManager.addfont(f)
            prop = fm.FontProperties(fname=f)
            plt.rcParams["font.family"] = prop.get_name()
            _FONT_PATH = f
            break
        except Exception:
            pass

if _FONT_PATH is None:
    try:
        plt.rcParams["font.family"] = "sans-serif"
    except Exception:
        pass


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_auc"], label="Val AUC", color="#2ca02c", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation AUC Over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_metrics_comparison(metrics_list, model_names, save_path=None):
    if not metrics_list:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = [5, 10, 20]
    hr_data = {}
    ndcg_data = {}
    for i, metrics in enumerate(metrics_list):
        hr_data[model_names[i]] = [metrics.get(f"hr@{k}", 0) for k in k_values]
        ndcg_data[model_names[i]] = [metrics.get(f"ndcg@{k}", 0) for k in k_values]

    x = np.arange(len(k_values))
    width = 0.2
    for i, (name, hr) in enumerate(hr_data.items()):
        axes[0].bar(x + i * width - width, hr, width, label=name)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("HR@K")
    axes[0].set_title("HR@K Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"@{k}" for k in k_values])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    for i, (name, ndcg) in enumerate(ndcg_data.items()):
        axes[1].bar(x + i * width - width, ndcg, width, label=name)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("NDCG@K")
    axes[1].set_title("NDCG@K Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"@{k}" for k in k_values])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_user_product_heatmap(scores_matrix, user_ids, product_names, save_path=None):
    display_users = min(15, len(user_ids))
    display_products = min(10, len(product_names))

    if len(scores_matrix) > display_users:
        indices = np.linspace(0, len(scores_matrix) - 1, display_users, dtype=int)
        scores_matrix = scores_matrix[indices]
        user_ids = [user_ids[i] for i in indices]
    if scores_matrix.shape[1] > display_products:
        p_indices = np.linspace(0, scores_matrix.shape[1] - 1, display_products, dtype=int)
        scores_matrix = scores_matrix[:, p_indices]
        product_names = [product_names[i] for i in p_indices]

    fig, ax = plt.subplots(figsize=(14, max(6, display_users * 0.4)))
    sns.heatmap(scores_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=product_names, yticklabels=user_ids,
                cbar_kws={"label": "Match Score"}, ax=ax)
    ax.set_xlabel("Insurance Products")
    ax.set_ylabel("Users")
    ax.set_title("User-Product Match Score Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_category_distribution(recommendation_results, save_path=None):
    from collections import Counter
    categories = [r["category"] for r in recommendation_results]
    cat_counts = Counter(categories)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cats = list(cat_counts.keys())
    counts = list(cat_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))

    axes[0].pie(counts, labels=cats, autopct="%1.1f%%", colors=colors)
    axes[0].set_title("Recommended Product Category Distribution")

    axes[1].barh(cats, counts, color=colors)
    axes[1].set_xlabel("Recommendation Count")
    axes[1].set_title("Category Recommendation Frequency")
    for i, v in enumerate(counts):
        axes[1].text(v + 0.5, i, str(v), va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_score_distribution(scores, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(scores, bins=30, color="#4c72b0", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Match Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Match Score Distribution")
    axes[0].axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
    axes[0].legend()

    from collections import Counter
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    binned = np.digitize(scores, bins[1:])
    binned_counts = Counter(binned)
    values = [binned_counts.get(i, 0) for i in range(len(labels))]
    axes[1].bar(labels, values, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"])
    axes[1].set_ylabel("Count")
    axes[1].set_title("Score Quality Distribution")
    for i, v in enumerate(values):
        axes[1].text(i, v + 1, str(v), ha="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_feature_importance(model, cat_names, num_names, save_path=None):
    import torch
    importance = {}
    if hasattr(model, "embeddings"):
        for name, emb in zip(cat_names, model.embeddings):
            importance[name] = torch.norm(emb.weight, dim=1).mean().item()

    if hasattr(model, "dnn"):
        first_linear = None
        for m in model.dnn:
            if isinstance(m, torch.nn.Linear):
                first_linear = m
                break
        if first_linear is not None:
            weights = first_linear.weight.data.abs().mean(dim=0).cpu().numpy()
            total_cat_dim = len(cat_names) * model.embed_dim
            cat_importance = weights[:total_cat_dim].reshape(len(cat_names), model.embed_dim).mean(axis=1)
            for name, ci in zip(cat_names, cat_importance):
                importance[name] = ci
            num_start = total_cat_dim
            for i, name in enumerate(num_names):
                if num_start + i < len(weights):
                    importance[name] = weights[num_start + i]

    if not importance:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_items = sorted(importance.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

    ax.barh(names, values, color=colors)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance Analysis (Derived from Model Weights)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
