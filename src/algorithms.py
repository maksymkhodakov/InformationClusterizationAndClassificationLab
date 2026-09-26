"""Обгортки над чотирма алгоритмами кластеризації + вимір часу й пам'яті.

Обрані алгоритми навмисно різнорідні за своєю природою:
    - KMeans          -- центроїдний, метричний (евклідова відстань до центра);
    - GaussianMixture  -- імовірнісний (EM), моделює кожен кластер еліпсоїдом
                          довільної орієнтації (коваріаційна матриця);
    - DBSCAN          -- густинний, не потребує апріорної кількості кластерів,
                          добре працює з кластерами довільної (неопуклої) форми;
    - AgglomerativeClustering (ward) -- ієрархічний, знизу-вгору.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass
class RunResult:
    algorithm: str
    labels: np.ndarray
    elapsed_sec: float
    peak_memory_kb: float


def _estimate_dbscan_eps(X: np.ndarray, k: int = 4) -> float:
    """Евристика к-найближчих сусідів для підбору eps у DBSCAN (метод «коліна»)."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, _ = nn.kneighbors(X)
    kth = np.sort(dist[:, -1])
    return float(np.percentile(kth, 97))


def run_kmeans(X: np.ndarray, n_clusters: int) -> RunResult:
    return _measured("KMeans", X, lambda: KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(X))


def run_gmm(X: np.ndarray, n_clusters: int) -> RunResult:
    return _measured(
        "GaussianMixture", X,
        lambda: GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=0).fit(X).predict(X),
    )


def run_dbscan(X: np.ndarray, n_clusters: int) -> RunResult:
    eps = _estimate_dbscan_eps(X)
    return _measured("DBSCAN", X, lambda: DBSCAN(eps=eps, min_samples=5).fit_predict(X))


def run_agglomerative(X: np.ndarray, n_clusters: int) -> RunResult:
    return _measured(
        "Agglomerative(Ward)", X,
        lambda: AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X),
    )


def _measured(name: str, X: np.ndarray, fn) -> RunResult:
    tracemalloc.start()
    t0 = time.perf_counter()
    labels = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return RunResult(algorithm=name, labels=np.asarray(labels), elapsed_sec=elapsed, peak_memory_kb=peak / 1024)


ALGORITHMS = {
    "KMeans": run_kmeans,
    "GaussianMixture": run_gmm,
    "DBSCAN": run_dbscan,
    "Agglomerative(Ward)": run_agglomerative,
}
