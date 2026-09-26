"""Допоміжні функції візуалізації наборів даних і результатів кластеризації."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

CMAP = "tab10"


def _to_2d(X: np.ndarray):
    if X.shape[1] == 2:
        return X, None
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X), pca


def plot_dataset(X, y_true, title, path):
    X2, _ = _to_2d(X)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X2[:, 0], X2[:, 1], c=y_true, cmap=CMAP, s=14, edgecolors="none")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_algorithm_grid(X, results: dict, dataset_title, path):
    """results: {algo_name: RunResult}"""
    X2, _ = _to_2d(X)
    n = len(results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(X2[:, 0], X2[:, 1], c=res.labels, cmap=CMAP, s=12, edgecolors="none")
        n_found = len(set(res.labels)) - (1 if -1 in res.labels else 0)
        ax.set_title(f"{name}\nt={res.elapsed_sec*1000:.1f} мс, "
                     f"пам'ять={res.peak_memory_kb:.0f} КБ, кластерів={n_found}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(dataset_title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_stability(X, labels_before, labels_after, moved_idx, title, path):
    X2, _ = _to_2d(X)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, labels, sub in zip(axes, [labels_before, labels_after], ["до δ-зсуву", "після δ-зсуву"]):
        ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=CMAP, s=12, edgecolors="none")
        ax.scatter(X2[moved_idx, 0], X2[moved_idx, 1], facecolors="none",
                   edgecolors="black", s=120, linewidths=1.5, label="перемiщена точка")
        ax.set_title(sub, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_bar(categories, values, ylabel, title, path, group_labels=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(categories))
    if group_labels is None:
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right")
    else:
        width = 0.8 / len(group_labels)
        for i, g in enumerate(group_labels):
            ax.bar(x + i * width, values[i], width=width, label=g)
        ax.set_xticks(x + width * (len(group_labels) - 1) / 2)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
