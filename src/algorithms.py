"""Чотири алгоритми кластеризації різних парадигм та вимірювання часу й пам'яті.

    KMeans              -- центроїдний (мінімізація внутрішньокластерної суми квадратів);
    GaussianMixture     -- імовірнісний (EM для суміші гаусіан з повними коваріаціями);
    DBSCAN              -- густинний (зв'язні області високої густини, виявляє шум);
    Agglomerative(Ward) -- ієрархічний висхідний (критерій Уорда).
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

ALGO_NAMES = ["KMeans", "GaussianMixture", "DBSCAN", "Agglomerative(Ward)"]


def dbscan_params(X: np.ndarray, percentile: float = 97.0) -> tuple[float, int]:
    """min_samples = max(5, 2d) (Sander et al., 1998); eps -- заданий процентиль
    відстаней до min_samples-го сусіда (наближення «коліна» k-distance графіка)."""
    min_samples = max(5, 2 * X.shape[1])
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    return float(np.percentile(dist[:, -1], percentile)), min_samples


def cluster(name: str, X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    if name == "KMeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    if name == "GaussianMixture":
        return GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed).fit(X).predict(X)
    if name == "DBSCAN":
        eps, ms = dbscan_params(X)
        return DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
    if name == "Agglomerative(Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    raise ValueError(name)


@dataclass
class Measurement:
    labels: np.ndarray
    time_ms_median: float
    time_ms_q1: float
    time_ms_q3: float
    peak_memory_kb: float


def measure(name: str, X: np.ndarray, k: int, repeats: int = 7) -> Measurement:
    """Час: 1 прогрівальний запуск + `repeats` вимірювань (медіана, квартилі) без
    tracemalloc, щоб трасування не спотворювало час. Пам'ять: окремий запуск під
    tracemalloc (пік виділень Python/NumPy за час роботи алгоритму)."""
    labels = cluster(name, X, k)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cluster(name, X, k)
        times.append((time.perf_counter() - t0) * 1000)
    tracemalloc.start()
    tracemalloc.reset_peak()
    cluster(name, X, k)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    return Measurement(np.asarray(labels), float(med), float(q1), float(q3), peak / 1024)
