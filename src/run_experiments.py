"""Головний скрипт дослідження: генерує всі набори даних, запускає на кожному
всі чотири алгоритми кластеризації, зберігає графіки (results/figures) та
підсумкові таблиці (results/tables) для звіту, а також виконує окремий
експеримент на стабільність (п.3 завдання)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import all_datasets
from src.algorithms import ALGORITHMS
from src.visualize import plot_dataset, plot_algorithm_grid, plot_stability, plot_bar
from src.stability import run_stability_experiment

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def safe_silhouette(X, labels):
    mask = labels != -1
    unique = set(labels[mask].tolist())
    if len(unique) < 2 or mask.sum() < len(unique) + 1:
        return np.nan
    try:
        return silhouette_score(X[mask], labels[mask])
    except Exception:
        return np.nan


def main():
    datasets = all_datasets()
    summary_rows = []

    for ds_name, (X, y_true, title) in datasets.items():
        n_clusters = len(set(y_true.tolist()))
        plot_dataset(X, y_true, f"{title}\n(вихідний набір, істинні мітки)",
                     os.path.join(FIG_DIR, f"{ds_name}__dataset.png"))

        results = {}
        for algo_name, run_fn in ALGORITHMS.items():
            res = run_fn(X, n_clusters)
            results[algo_name] = res
            ari = adjusted_rand_score(y_true, res.labels)
            sil = safe_silhouette(X, res.labels)
            n_found = len(set(res.labels.tolist())) - (1 if -1 in res.labels else 0)
            summary_rows.append({
                "dataset": ds_name,
                "dataset_title": title,
                "algorithm": algo_name,
                "n_points": len(X),
                "n_dims": X.shape[1],
                "n_clusters_true": n_clusters,
                "n_clusters_found": n_found,
                "ARI": round(ari, 4),
                "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan,
                "time_ms": round(res.elapsed_sec * 1000, 3),
                "peak_memory_KB": round(res.peak_memory_kb, 2),
            })

        plot_algorithm_grid(X, results, title, os.path.join(FIG_DIR, f"{ds_name}__algorithms.png"))
        print(f"[ok] {ds_name}: {title}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(TAB_DIR, "summary.csv"), index=False)

    # Зведені графіки часу й пам'яті (усереднено по наборах даних для кожного алгоритму)
    agg = summary.groupby("algorithm")[["time_ms", "peak_memory_KB"]].mean().reindex(list(ALGORITHMS.keys()))
    plot_bar(agg.index.tolist(), agg["time_ms"].tolist(), "Середній час, мс",
             "Середній час виконання алгоритму (усереднено по всіх наборах даних)",
             os.path.join(FIG_DIR, "summary__time.png"))
    plot_bar(agg.index.tolist(), agg["peak_memory_KB"].tolist(), "Пікова пам'ять, КБ",
             "Середній обсяг пам'яті алгоритму (усереднено по всіх наборах даних)",
             os.path.join(FIG_DIR, "summary__memory.png"))

    ari_pivot = summary.pivot(index="dataset", columns="algorithm", values="ARI")
    ari_pivot.to_csv(os.path.join(TAB_DIR, "ari_pivot.csv"))

    run_stability(datasets)


def run_stability(datasets):
    stability_rows = []
    cases = [
        ("circles_det", "KMeans"),
        ("circles_det", "DBSCAN"),
        ("letters", "DBSCAN"),
        ("letters", "KMeans"),
        ("mixture", "GaussianMixture"),
    ]
    deltas = [0.01, 0.1, 0.5, 1.5]

    for ds_name, algo_name in cases:
        X, y_true, title = datasets[ds_name]
        n_clusters = len(set(y_true.tolist()))

        cluster0_idx = np.where(y_true == 0)[0]
        centroid = X[cluster0_idx].mean(axis=0)
        dists = np.linalg.norm(X[cluster0_idx] - centroid, axis=1)
        interior_idx = cluster0_idx[np.argmin(dists)]
        boundary_idx = cluster0_idx[np.argmax(dists)]
        point_indices = [int(interior_idx), int(boundary_idx)]

        rows = run_stability_experiment(X, n_clusters, algo_name, point_indices, deltas)
        for r in rows:
            role = "внутрішня" if r["point_idx"] == interior_idx else "межова"
            stability_rows.append({
                "dataset": ds_name, "algorithm": algo_name, "point_role": role,
                "point_idx": r["point_idx"], "delta": r["delta"],
                "ARI_before_after": round(r["ari"], 4), "n_labels_changed": r["n_labels_changed"],
            })
            if r["delta"] in (0.1, 1.5):
                X2 = X.copy()
                fig_path = os.path.join(
                    FIG_DIR, f"stability__{ds_name}__{algo_name}__{role}__d{r['delta']}.png")
                plot_stability(X, r["labels_before"], r["labels_after"], r["point_idx"],
                                f"{title}\n{algo_name}, точка={role}, δ={r['delta']}, ARI={r['ari']:.3f}",
                                fig_path)
        print(f"[ok] стабільність: {ds_name} / {algo_name}")

    stab_df = pd.DataFrame(stability_rows)
    stab_df.to_csv(os.path.join(TAB_DIR, "stability.csv"), index=False)


if __name__ == "__main__":
    main()
