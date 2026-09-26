"""Дослідження алгоритмів на «неперервність» (стабільність), п.3 завдання.

Точку (або набір точок) набору Д зміщують на відстань δ у випадковому напрямку,
алгоритм застосовують повторно до Д' і порівнюють розбиття. Номери кластерів
довільні (алгоритм може перенумерувати кластери), тому перед підрахунком змінених
міток розбиття «після» зіставляється з розбиттям «до» угорським алгоритмом.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .algorithms import cluster


def align_labels(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Перенумеровує кластери `after` так, щоб максимально збігатися з `before`.
    Шум DBSCAN (-1) не перенумеровується; незіставлені кластери отримують нові номери."""
    b_ids = [c for c in np.unique(before) if c != -1]
    a_ids = [c for c in np.unique(after) if c != -1]
    mapping = {-1: -1}
    if b_ids and a_ids:
        overlap = np.array([[np.sum((after == a) & (before == b)) for b in b_ids] for a in a_ids])
        rows, cols = linear_sum_assignment(-overlap)
        mapping.update({a_ids[r]: b_ids[c] for r, c in zip(rows, cols)})
    next_id = max(b_ids, default=-1) + 1
    for a in a_ids:
        if a not in mapping:
            mapping[a] = next_id
            next_id += 1
    return np.array([mapping[a] for a in after])


def interior_and_boundary_points(X: np.ndarray, y: np.ndarray, cluster_id: int = 0) -> tuple[int, int]:
    """Внутрішня точка -- найближча до центроїда кластера; межова -- точка кластера,
    найближча до будь-якої точки іншого кластера (лежить на межі між кластерами)."""
    idx = np.where(y == cluster_id)[0]
    other = np.where(y != cluster_id)[0]
    interior = idx[np.argmin(np.linalg.norm(X[idx] - X[idx].mean(axis=0), axis=1))]
    d = np.linalg.norm(X[idx][:, None, :] - X[other][None, :, :], axis=2).min(axis=1)
    boundary = idx[np.argmin(d)]
    return int(interior), int(boundary)


def perturb(X: np.ndarray, indices: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(len(indices), X.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    X2 = X.copy()
    X2[indices] += delta * directions
    return X2


def compare(before: np.ndarray, after: np.ndarray, moved: np.ndarray) -> dict:
    aligned = align_labels(before, after)
    changed = aligned != before
    mask_other = np.ones(len(before), bool)
    mask_other[moved] = False
    n_before = len(set(before.tolist()) - {-1})
    n_after = len(set(after.tolist()) - {-1})
    return {
        "ARI": adjusted_rand_score(before, after),
        "changed_total": int(changed.sum()),
        "changed_moved": int(changed[moved].sum()),
        "changed_other": int(changed[mask_other].sum()),
        "k_changed": int(n_before != n_after),
        "aligned_after": aligned,
    }


FIXED_EPS_DBSCAN = "DBSCAN(фікс. eps)"


def make_clusterer(algo, X, k):
    """Для FIXED_EPS_DBSCAN параметри eps/min_samples оцінюються один раз на вихідному
    наборі й не перераховуються для Д' -- це відокремлює чутливість самого DBSCAN від
    чутливості процедури автоматичного вибору eps."""
    if algo == FIXED_EPS_DBSCAN:
        from sklearn.cluster import DBSCAN
        from .algorithms import dbscan_params
        eps, ms = dbscan_params(X)
        return lambda Z: DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
    return lambda Z: cluster(algo, Z, k)


def run_stability(X, y, algo, deltas, scenarios, repeats=5, seed=0):
    """scenarios: список (назва, функція rng -> масив індексів зміщуваних точок)."""
    k = len(np.unique(y))
    fit = make_clusterer(algo, X, k)
    base = fit(X)
    rows, examples = [], {}
    for sc_name, pick in scenarios:
        for delta in deltas:
            for rep in range(repeats):
                rng = np.random.default_rng([seed, rep, int(delta * 1000)])
                moved = np.atleast_1d(pick(rng))
                X2 = perturb(X, moved, delta, rng)
                after = fit(X2)
                res = compare(base, after, moved)
                rows.append({"algorithm": algo, "scenario": sc_name, "delta": delta, "repeat": rep,
                             **{k_: v for k_, v in res.items() if k_ != "aligned_after"}})
                prev = examples.get((sc_name, delta))
                if prev is None or res["ARI"] < prev[3]:
                    examples[(sc_name, delta)] = (X2, moved, res["aligned_after"], res["ARI"])
    return base, rows, examples


def init_stability(X, y, algo, seeds=range(20)):
    """Стабільність відносно ініціалізації: попарний ARI між розбиттями при різних seed
    (для KMeans використовується n_init=1, щоб побачити «чисту» залежність від старту)."""
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    k = len(np.unique(y))
    labels = []
    for s in seeds:
        if algo == "KMeans":
            labels.append(KMeans(n_clusters=k, n_init=1, init="random", random_state=s).fit_predict(X))
        else:
            labels.append(GaussianMixture(n_components=k, covariance_type="full", n_init=1,
                                          init_params="random_from_data", random_state=s).fit(X).predict(X))
    aris = [adjusted_rand_score(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    truth = [adjusted_rand_score(y, l) for l in labels]
    return np.array(aris), np.array(truth)
