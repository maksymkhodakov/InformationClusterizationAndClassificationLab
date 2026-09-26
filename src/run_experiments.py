"""Відтворює всі результати звіту: python -m src.run_experiments

results/figures -- рисунки, results/tables -- таблиці (CSV)."""
from __future__ import annotations

import platform
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .algorithms import ALGO_NAMES, cluster, measure
from .datasets import FAMILIES, GENERATORS, all_datasets, uniform_in_ball
from .stability import FIXED_EPS_DBSCAN, init_stability, interior_and_boundary_points, run_stability
from . import visualize as viz

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

DELTAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
STABILITY_DATASETS = ["circles_det", "ellipses_nonparallel", "letters_gst", "rings", "real_iris"]
STABILITY_EXAMPLES = [  # (набір, алгоритм, сценарій, δ) -- випадки, що ілюструються рисунками у звіті
    ("letters_gst", "DBSCAN", "межова точка", 0.05),
    ("letters_gst", FIXED_EPS_DBSCAN, "межова точка", 0.05),
    ("circles_det", "DBSCAN", "10% точок", 1.0),
    ("rings", "GaussianMixture", "внутрішня точка", 0.25),
    ("rings", "Agglomerative(Ward)", "1% точок", 0.01),
    ("real_iris", "GaussianMixture", "10% точок", 1.0),
]


def silhouette(X, labels):
    mask = labels != -1
    if len(set(labels[mask].tolist())) < 2:
        return np.nan
    return silhouette_score(X[mask], labels[mask])


def environment():
    rows = {
        "Python": sys.version.split()[0], "NumPy": np.__version__, "scikit-learn": sklearn.__version__,
        "SciPy": scipy.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "ОС": f"{platform.system()} {platform.release()}", "Процесор": platform.processor() or platform.machine(),
    }
    pd.DataFrame(rows.items(), columns=["component", "version"]).to_csv(TAB / "environment.csv", index=False)


def main_comparison(datasets):
    rows, meta = [], []
    for ds in datasets.values():
        meta.append({"dataset": ds.key, "title": ds.title, "shape": ds.shape, "orientation": ds.orientation,
                     "generation": ds.generation, "n": len(ds.X), "d": ds.X.shape[1], "k": ds.n_clusters})
        viz.plot_dataset(ds, FIG / f"{ds.key}__dataset.png")
        results = {}
        for algo in ALGO_NAMES:
            m = measure(algo, ds.X, ds.n_clusters)
            results[algo] = m
            rows.append({
                "dataset": ds.key, "algorithm": algo, "k_true": ds.n_clusters,
                "k_found": len(set(m.labels.tolist()) - {-1}), "noise": int((m.labels == -1).sum()),
                "ARI": adjusted_rand_score(ds.y, m.labels), "silhouette": silhouette(ds.X, m.labels),
                "time_ms": m.time_ms_median, "time_q1": m.time_ms_q1, "time_q3": m.time_ms_q3,
                "peak_memory_KB": m.peak_memory_kb,
            })
        viz.plot_algorithm_grid(ds, results, FIG / f"{ds.key}__algorithms.png")
        if ds.key in ("ellipses_nonparallel", "rings", "real_iris", "real_wine", "letters_pex"):
            viz.plot_contingency(ds, results, FIG / f"{ds.key}__contingency.png")
        print(f"[ok] {ds.key}")
    pd.DataFrame(meta).to_csv(TAB / "datasets.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB / "summary.csv", index=False)
    order = list(datasets)
    ari = summary.pivot(index="dataset", columns="algorithm", values="ARI").loc[order, ALGO_NAMES]
    ari.to_csv(TAB / "ari_pivot.csv")
    viz.plot_heatmap(ari, FIG / "summary__ari_heatmap.png", "ARI (відповідність істинному розбиттю)")
    sil = summary.pivot(index="dataset", columns="algorithm", values="silhouette").loc[order, ALGO_NAMES]
    viz.plot_heatmap(sil, FIG / "summary__silhouette_heatmap.png", "Silhouette (внутрішня якість)", vmin=-0.2, vmax=1)
    viz.plot_perf_bars(summary, FIG / "summary__perf.png")
    return summary


def input_visualizations(datasets):
    viz.plot_input_methods(datasets["letters_pex"], FIG / "input__methods_letters_pex.png")
    viz.plot_input_methods(datasets["rings"], FIG / "input__methods_rings.png")
    from sklearn.datasets import load_iris
    viz.plot_pairplot(datasets["real_iris"], load_iris().feature_names, FIG / "input__pairplot_iris.png")


def intermediate_visualizations(datasets):
    info = {}
    info["kmeans_iterations"] = viz.plot_kmeans_iterations(
        datasets["ellipses_nonparallel"].X, 4, FIG / "inter__kmeans_iterations.png")
    info["gmm_iterations"] = viz.plot_gmm_iterations(
        datasets["ellipses_nonparallel"].X, 4, FIG / "inter__gmm_iterations.png")
    viz.plot_dendrogram(datasets["mixture"].X, 5, FIG / "inter__dendrogram_mixture.png",
                        "Дендрограма (метод Уорда), набір «Суміш типів» — останні 40 об'єднань")
    for key in ("letters_gst", "rings"):
        eps, ms, core, border, noise = viz.plot_dbscan_diagnostics(
            datasets[key].X, FIG / f"inter__dbscan_{key}.png", f"DBSCAN, набір «{datasets[key].title}»")
        info[f"dbscan_{key}"] = f"eps={eps:.3f}; min_samples={ms}; core={core}; border={border}; noise={noise}"
    pd.DataFrame(info.items(), columns=["item", "value"]).to_csv(TAB / "intermediate.csv", index=False)


def robustness():
    rows = []
    for fam, factory in FAMILIES.items():
        for gen in GENERATORS:
            for seed in range(100, 105):
                ds = factory(seed, gen)
                for algo in ALGO_NAMES:
                    rows.append({"family": fam, "generator": gen, "seed": seed, "algorithm": algo,
                                 "ARI": adjusted_rand_score(ds.y, cluster(algo, ds.X, ds.n_clusters))})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "robustness_raw.csv", index=False)
    agg = df.groupby(["family", "algorithm"], sort=False)["ARI"].agg(["mean", "std", "min"]).reset_index()
    agg.to_csv(TAB / "robustness.csv", index=False)
    gen_agg = df.groupby(["generator", "algorithm"], sort=False)["ARI"].mean().unstack()[ALGO_NAMES]
    gen_agg.to_csv(TAB / "robustness_by_generator.csv")
    viz.plot_robustness(df, FIG / "summary__robustness.png")
    print("[ok] robustness")


def initialization(datasets):
    rows = []
    for key in ("ellipses_nonparallel", "letters_gst", "rings", "real_iris"):
        ds = datasets[key]
        for algo in ("KMeans", "GaussianMixture"):
            pair, truth = init_stability(ds.X, ds.y, algo)
            rows.append({"dataset": key, "algorithm": algo, "pairwise_ARI_mean": pair.mean(),
                         "pairwise_ARI_min": pair.min(), "truth_ARI_mean": truth.mean(),
                         "truth_ARI_min": truth.min(), "truth_ARI_max": truth.max()})
    pd.DataFrame(rows).to_csv(TAB / "init_stability.csv", index=False)
    print("[ok] initialization")


def scalability():
    rows = []
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], float)
    for n in (250, 500, 1000, 2000, 4000, 8000):
        X = np.concatenate([c + 2 * uniform_in_ball(n // 4, 2, rng) for c in centers])
        for algo in ALGO_NAMES:
            m = measure(algo, X, 4, repeats=3)
            rows.append({"n": n, "algorithm": algo, "time_ms": m.time_ms_median, "peak_memory_KB": m.peak_memory_kb})
        print(f"[ok] scalability n={n}")
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "scalability.csv", index=False)
    viz.plot_scalability(df, FIG / "summary__scalability.png")


def eps_threshold(X, X2, n_ex):
    """Чи пояснюється зміна розбиття DBSCAN зміною автоматично обраного eps: eps до/після
    зсуву та найближче до eps0 значення, за якого змінюється кількість кластерів на X."""
    from sklearn.cluster import DBSCAN
    from .algorithms import dbscan_params
    eps0, ms = dbscan_params(X)
    eps1, _ = dbscan_params(X2)
    k_of = lambda e: len(set(DBSCAN(eps=e, min_samples=ms).fit_predict(X).tolist()) - {-1})
    k0 = k_of(eps0)
    grid = np.arange(eps0 * 0.9, eps0 * 1.1, 1e-4)
    changes = [e for e in grid if k_of(e) != k0]
    critical = min(changes, key=lambda e: abs(e - eps0)) if changes else np.nan
    return {"example": n_ex, "eps_before": eps0, "eps_after": eps1, "critical_eps": critical, "k_at_eps_before": k0,
            "k_at_critical": k_of(critical) if changes else np.nan}


def stability(datasets):
    raw = []
    shown = []
    eps_rows = []
    for key in STABILITY_DATASETS:
        ds = datasets[key]
        interior, boundary = interior_and_boundary_points(ds.X, ds.y)
        n = len(ds.X)
        scenarios = [
            ("внутрішня точка", lambda rng, i=interior: np.array([i])),
            ("межова точка", lambda rng, b=boundary: np.array([b])),
            ("1% точок", lambda rng: rng.choice(n, max(1, n // 100), replace=False)),
            ("5% точок", lambda rng: rng.choice(n, n // 20, replace=False)),
            ("10% точок", lambda rng: rng.choice(n, n // 10, replace=False)),
        ]
        for algo in ALGO_NAMES + [FIXED_EPS_DBSCAN]:
            base, rows, examples = run_stability(ds.X, ds.y, algo, DELTAS, scenarios)
            for r in rows:
                r["dataset"] = key
            raw.extend(rows)
            for n_ex, (d_key, d_algo, sc, delta) in enumerate(STABILITY_EXAMPLES, start=1):
                if (d_key, d_algo) != (key, algo):
                    continue
                X2, moved, aligned, ari = examples[(sc, delta)]
                path = FIG / f"stab_example_{n_ex}.png"
                viz.plot_stability_example(ds.X, X2, base, aligned, moved,
                                           f"{ds.title} — {algo}, «{sc}», δ={delta}: ARI={ari:.3f}", path)
                shown.append({"n": n_ex, "dataset": key, "algorithm": algo, "scenario": sc, "delta": delta,
                              "ARI": ari, "changed": int((aligned != base).sum()), "figure": path.name})
                if algo == "DBSCAN":
                    eps_rows.append(eps_threshold(ds.X, X2, n_ex))
        print(f"[ok] stability {key}")
    df = pd.DataFrame(raw)
    df.to_csv(TAB / "stability_raw.csv", index=False)
    agg = df.groupby(["dataset", "algorithm", "scenario", "delta"], sort=False).agg(
        ARI_mean=("ARI", "mean"), ARI_min=("ARI", "min"),
        changed_other_mean=("changed_other", "mean"), changed_other_max=("changed_other", "max"),
        changed_moved_mean=("changed_moved", "mean"), k_changed_rate=("k_changed", "mean"),
    ).reset_index()
    agg.to_csv(TAB / "stability.csv", index=False)
    pd.DataFrame(shown).sort_values("n").to_csv(TAB / "stability_examples.csv", index=False)
    pd.DataFrame(eps_rows).to_csv(TAB / "eps_threshold.csv", index=False)
    for key in STABILITY_DATASETS:
        viz.plot_stability_curves(agg[agg["dataset"] == key], datasets[key].title, FIG / f"stab_curves__{key}.png")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for f in list(FIG.glob("*.png")) + list(TAB.glob("*.csv")):
        f.unlink()
    environment()
    datasets = all_datasets()
    main_comparison(datasets)
    input_visualizations(datasets)
    intermediate_visualizations(datasets)
    robustness()
    initialization(datasets)
    stability(datasets)
    scalability()


if __name__ == "__main__":
    main()
