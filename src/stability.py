"""Дослідження алгоритмів на "неперервність" (стабільність) за завданням п.3:
невелике (δ) зміщення однієї точки набору й повторний запуск алгоритму,
порівняння отриманого розбиття на кластери з вихідним."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .algorithms import ALGORITHMS


def perturb_point(X: np.ndarray, idx: int, delta: float, direction: np.ndarray = None, seed: int = 0) -> np.ndarray:
    """Повертає копію X, де точка idx зміщена на відстань delta у напрямку direction
    (якщо не задано -- випадковий одиничний напрямок)."""
    rng = np.random.default_rng(seed)
    if direction is None:
        direction = rng.normal(size=X.shape[1])
        direction /= np.linalg.norm(direction)
    X2 = X.copy()
    X2[idx] = X2[idx] + delta * direction
    return X2


def run_stability_experiment(X, n_clusters, algo_name, point_indices, deltas, seed=0):
    """Для заданого алгоритму, набору "цікавих" індексів точок (напр. в центрі
    та на межі кластера) і списку значень delta прогонити алгоритм до і після
    зсуву, порахувати ARI (Adjusted Rand Index) між розбиттями."""
    run_fn = ALGORITHMS[algo_name]
    base = run_fn(X, n_clusters)
    rows = []
    for idx in point_indices:
        for delta in deltas:
            X2 = perturb_point(X, idx, delta, seed=seed)
            after = run_fn(X2, n_clusters)
            ari = adjusted_rand_score(base.labels, after.labels)
            n_changed = int(np.sum(base.labels != after.labels))
            rows.append({
                "algorithm": algo_name,
                "point_idx": idx,
                "delta": delta,
                "ari": ari,
                "n_labels_changed": n_changed,
                "labels_before": base.labels,
                "labels_after": after.labels,
            })
    return rows
