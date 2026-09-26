"""Візуалізація вихідних даних, проміжних і остаточних результатів кластеризації."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .algorithms import dbscan_params

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 100})
PALETTE = plt.get_cmap("tab10").colors
NOISE = "#9e9e9e"
ALGO_COLORS = {"KMeans": "#1f77b4", "GaussianMixture": "#ff7f0e", "DBSCAN": "#2ca02c", "Agglomerative(Ward)": "#d62728"}
EXTRA_COLORS = {"DBSCAN(фікс. eps)": "#8c564b"}


def colors_for(labels):
    noise = matplotlib.colors.to_rgba(NOISE)
    return [noise if l == -1 else matplotlib.colors.to_rgba(PALETTE[int(l) % 10]) for l in labels]


def project(X):
    return X if X.shape[1] <= 3 else PCA(n_components=2, random_state=0).fit_transform(X)


def scatter(ax, X, labels, title="", s=9):
    P = project(X)
    c = colors_for(labels) if labels is not None else "#555555"
    if P.shape[1] == 3:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, s=s, depthshade=False, linewidths=0)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    else:
        ax.scatter(P[:, 0], P[:, 1], c=c, s=s, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("PC1" if X.shape[1] > 3 else "x1")
        ax.set_ylabel("PC2" if X.shape[1] > 3 else "x2")
    ax.set_title(title)


def new_axes(fig, nrows, ncols, i, dim):
    return fig.add_subplot(nrows, ncols, i, projection="3d" if dim == 3 else None)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ вихідні дані
def plot_dataset(ds, path):
    dim = min(ds.X.shape[1], 3)
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = new_axes(fig, 1, 1, 1, dim)
    scatter(ax, ds.X, ds.y, f"{ds.title}\nn={len(ds.X)}, d={ds.X.shape[1]}, k={ds.n_clusters}")
    save(fig, path)


def plot_input_methods(ds, path):
    """Чотири способи подати вихідний (немаркований) набір: точкова діаграма,
    гексагональна густина, маргінальні гістограми, ізолінії KDE."""
    X = ds.X
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].scatter(X[:, 0], X[:, 1], s=6, c="#444444", linewidths=0)
    axes[0].set_title("Діаграма розсіювання (без міток)")
    hb = axes[1].hexbin(X[:, 0], X[:, 1], gridsize=28, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=axes[1], label="к-ть точок")
    axes[1].set_title("Гексагональна діаграма густини")
    axes[2].hist(X[:, 0], bins=40, alpha=0.6, label="x1")
    axes[2].hist(X[:, 1], bins=40, alpha=0.6, label="x2")
    axes[2].legend(); axes[2].set_title("Маргінальні гістограми ознак")
    kde = gaussian_kde(X.T)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 120),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 120))
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    axes[3].contourf(gx, gy, z, levels=14, cmap="magma")
    axes[3].set_title("Ядерна оцінка густини (KDE)")
    for ax in (axes[0], axes[1], axes[3]):
        ax.set_aspect("equal", adjustable="datalim")
    save(fig, path)


def plot_pairplot(ds, feature_names, path):
    X, y = ds.X, ds.y
    d = X.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.1 * d, 2.1 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                for c in np.unique(y):
                    ax.hist(X[y == c, i], bins=15, alpha=0.6, color=PALETTE[c])
            else:
                ax.scatter(X[:, j], X[:, i], c=colors_for(y), s=5, linewidths=0)
            if i == d - 1:
                ax.set_xlabel(feature_names[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(feature_names[i], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{ds.title}: матриця діаграм розсіювання (колір — істинний клас)")
    save(fig, path)


# ------------------------------------------------------------------ остаточні результати
def plot_algorithm_grid(ds, results, path):
    dim = min(ds.X.shape[1], 3)
    fig = plt.figure(figsize=(9, 8.6))
    for i, (name, m) in enumerate(results.items(), start=1):
        ax = new_axes(fig, 2, 2, i, dim)
        k_found = len(set(m.labels.tolist()) - {-1})
        noise = int((m.labels == -1).sum())
        extra = f", шум={noise}" if noise else ""
        scatter(ax, ds.X, m.labels, f"{name}: кластерів={k_found}{extra}\n"
                                    f"t={m.time_ms_median:.1f} мс, пам'ять={m.peak_memory_kb:.0f} КБ")
    fig.suptitle(ds.title, fontsize=11)
    save(fig, path)


def plot_contingency(ds, results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.2))
    for ax, (name, m) in zip(axes, results.items()):
        ct = pd.crosstab(ds.y, m.labels)
        cm = ct.values
        ax.imshow(cm, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=7, color="white" if v > cm.max() / 2 else "black")
        ax.set_title(name); ax.set_xlabel("знайдений кластер (−1 = шум)"); ax.set_ylabel("істинний клас")
        ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(ct.columns, fontsize=7)
        ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(ct.index, fontsize=7)
    fig.suptitle(f"{ds.title}: таблиці спряженості «істинний клас × знайдений кластер»")
    save(fig, path)


def plot_heatmap(df, path, title, fmt="{:.2f}", cmap="RdYlGn", vmin=0, vmax=1):
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(df) + 1.5))
    im = ax.imshow(df.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=20, ha="right")
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
    for (i, j), v in np.ndenumerate(df.values):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    save(fig, path)


# ------------------------------------------------------------------ проміжні результати
def _ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for nsig in (1, 2):
        w, h = 2 * nsig * np.sqrt(vals[::-1])
        ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False, color=color, lw=1.4, alpha=0.9 if nsig == 1 else 0.5))


def plot_kmeans_iterations(X, k, path, seed=3):
    """Власна покрокова реалізація алгоритму Ллойда (для показу проміжних станів)."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    history = [C.copy()]
    labels_hist = []
    for _ in range(100):
        labels = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        labels_hist.append(labels)
        C_new = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        history.append(C_new.copy())
        if np.allclose(C_new, C):
            break
        C = C_new
    n_it = len(labels_hist)
    shown = sorted(set([0, 1, 2, n_it - 1]))
    fig, axes = plt.subplots(1, len(shown), figsize=(3.8 * len(shown), 3.8))
    H = np.array(history)
    for ax, it in zip(axes, shown):
        ax.scatter(X[:, 0], X[:, 1], c=colors_for(labels_hist[it]), s=6, linewidths=0)
        for j in range(k):
            ax.plot(H[: it + 2, j, 0], H[: it + 2, j, 1], "-o", color="black", ms=3, lw=1)
        ax.scatter(H[it + 1, :, 0], H[it + 1, :, 1], marker="X", s=120, c="black", edgecolors="white")
        ax.set_title(f"ітерація {it + 1}" + (" (збіжність)" if it == n_it - 1 else ""))
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("K-Means: призначення точок і траєкторії центроїдів на проміжних ітераціях (випадкова ініціалізація)")
    save(fig, path)
    return n_it


def plot_gmm_iterations(X, k, path, seed=1):
    gm = GaussianMixture(n_components=k, covariance_type="full", max_iter=1, warm_start=True,
                         init_params="random_from_data", random_state=seed)
    snaps, it, prev = {}, 0, None
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for it in range(1, 301):
            gm.fit(X)
            if it in (1, 3, 10):
                snaps[it] = (gm.means_.copy(), gm.covariances_.copy(), gm.predict(X))
            if prev is not None and abs(gm.lower_bound_ - prev) < 1e-5:
                break
            prev = gm.lower_bound_
    snaps[it] = (gm.means_.copy(), gm.covariances_.copy(), gm.predict(X))
    fig, axes = plt.subplots(1, len(snaps), figsize=(3.8 * len(snaps), 3.8))
    for ax, (i, (mu, cov, lab)) in zip(axes, sorted(snaps.items())):
        ax.scatter(X[:, 0], X[:, 1], c=colors_for(lab), s=5, linewidths=0, alpha=0.7)
        for j in range(k):
            _ellipse(ax, mu[j], cov[j], PALETTE[j % 10])
            ax.plot(*mu[j], "k+", ms=10)
        ax.set_title(f"EM-ітерація {i}" + (" (збіжність)" if i == max(snaps) else ""))
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("GMM: еволюція компонент (еліпси 1σ і 2σ) на ітераціях EM-алгоритму")
    save(fig, path)
    return it


def plot_dendrogram(X, k, path, title):
    Z = linkage(X, method="ward")
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2
    fig, ax = plt.subplots(figsize=(10, 4.2))
    dendrogram(Z, truncate_mode="lastp", p=40, color_threshold=cut, ax=ax, leaf_font_size=7)
    ax.axhline(cut, ls="--", color="black", lw=1)
    ax.text(ax.get_xlim()[1], cut, f"  зріз → {k} кластери", va="bottom", ha="right")
    ax.set_ylabel("відстань об'єднання (критерій Уорда)")
    ax.set_title(title)
    save(fig, path)
    return Z


def plot_dbscan_diagnostics(X, path, title):
    eps, ms = dbscan_params(X)
    dist, _ = NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)
    kd = np.sort(dist[:, -1])
    db = DBSCAN(eps=eps, min_samples=ms).fit(X)
    core = np.zeros(len(X), bool)
    core[db.core_sample_indices_] = True
    noise = db.labels_ == -1
    border = ~core & ~noise
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(kd, lw=1.5)
    axes[0].axhline(eps, color="red", ls="--", label=f"eps = {eps:.3f} (97-й процентиль)")
    axes[0].set_xlabel("точки, упорядковані за зростанням"); axes[0].set_ylabel(f"відстань до {ms}-го сусіда")
    axes[0].set_title("k-distance графік для вибору eps"); axes[0].legend()
    c = np.array(colors_for(db.labels_))
    axes[1].scatter(X[core, 0], X[core, 1], c=c[core], s=10, linewidths=0, label="кореневі")
    axes[1].scatter(X[border, 0], X[border, 1], facecolors="none", edgecolors=c[border], s=22, label="межові")
    axes[1].scatter(X[noise, 0], X[noise, 1], c="black", marker="x", s=22, label="шум")
    axes[1].legend(fontsize=7); axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].set_title(f"Типи точок DBSCAN: кореневих {core.sum()}, межових {border.sum()}, шуму {noise.sum()}")
    fig.suptitle(title)
    save(fig, path)
    return eps, ms, int(core.sum()), int(border.sum()), int(noise.sum())


# ------------------------------------------------------------------ стабільність
def plot_stability_example(X, X2, before, after_aligned, moved, title, path):
    changed = np.where(after_aligned != before)[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, P, lab, sub in ((axes[0], X, before, "до δ-зсуву"), (axes[1], X2, after_aligned, "після δ-зсуву")):
        P2 = project(P)[:, :2]
        ax.scatter(P2[:, 0], P2[:, 1], c=colors_for(lab), s=8, linewidths=0)
        ax.scatter(P2[moved, 0], P2[moved, 1], facecolors="none", edgecolors="black", s=110, lw=1.3,
                   label="зміщені точки")
        if sub.startswith("після") and len(changed):
            ax.scatter(P2[changed, 0], P2[changed, 1], facecolors="none", edgecolors="red", s=45, lw=1,
                       label=f"змінили кластер ({len(changed)})")
        ax.set_title(sub); ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=7, loc="best")
    if X.shape[1] == 2:
        for i in moved:
            axes[1].annotate("", xy=X2[i], xytext=X[i], arrowprops=dict(arrowstyle="->", color="black", lw=1))
    fig.suptitle(title)
    save(fig, path)


def plot_stability_curves(df, dataset_title, path, metric="ARI_mean", ylabel="середній ARI (до/після)"):
    scenarios = list(dict.fromkeys(df["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.3 * len(scenarios), 3.4), sharey=True)
    for ax, sc in zip(axes, scenarios):
        sub = df[df["scenario"] == sc]
        for algo, g in sub.groupby("algorithm", sort=False):
            ax.plot(g["delta"], g[metric], "--s" if algo in EXTRA_COLORS else "-o", ms=3, label=algo,
                    color={**ALGO_COLORS, **EXTRA_COLORS}[algo])
        ax.set_xscale("log"); ax.set_title(sc); ax.set_xlabel("δ"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"Стабільність: {dataset_title}")
    save(fig, path)


# ------------------------------------------------------------------ продуктивність
def plot_perf_bars(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    datasets = list(dict.fromkeys(df["dataset"]))
    algos = list(ALGO_COLORS)
    w = 0.8 / len(algos)
    x = np.arange(len(datasets))
    for i, a in enumerate(algos):
        g = df[df["algorithm"] == a].set_index("dataset").loc[datasets]
        err = [g["time_ms"] - g["time_q1"], g["time_q3"] - g["time_ms"]]
        axes[0].bar(x + i * w, g["time_ms"], w, yerr=err, label=a, color=ALGO_COLORS[a], capsize=1.5)
        axes[1].bar(x + i * w, g["peak_memory_KB"], w, label=a, color=ALGO_COLORS[a])
    for ax, yl, t in ((axes[0], "час, мс (медіана, IQR)", "Час виконання"), (axes[1], "пікова пам'ять, КБ", "Обсяг пам'яті")):
        ax.set_xticks(x + w * 1.5); ax.set_xticklabels(datasets, rotation=55, ha="right", fontsize=7)
        ax.set_yscale("log"); ax.set_ylabel(yl); ax.set_title(t); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    save(fig, path)


def plot_scalability(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, g in df.groupby("algorithm", sort=False):
        axes[0].plot(g["n"], g["time_ms"], "-o", ms=3, label=a, color=ALGO_COLORS[a])
        axes[1].plot(g["n"], g["peak_memory_KB"] / 1024, "-o", ms=3, label=a, color=ALGO_COLORS[a])
    n = np.array(sorted(df["n"].unique()), float)
    for ax, col, lab in ((axes[0], "time_ms", "час, мс"), (axes[1], "peak_memory_KB", "пам'ять, МБ")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("кількість точок n"); ax.set_ylabel(lab)
        ax.set_xticks(n); ax.set_xticklabels([str(int(v)) for v in n])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=0.3, which="both")
    ref = df[df["algorithm"] == "Agglomerative(Ward)"]["peak_memory_KB"].iloc[-1] / 1024
    axes[1].plot(n, ref * (n / n[-1]) ** 2, "k:", lw=1, label="~n²")
    axes[1].plot(n, ref * (n / n[-1]), "k--", lw=1, label="~n")
    axes[0].set_title("Масштабованість за часом"); axes[1].set_title("Масштабованість за пам'яттю")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    save(fig, path)


def plot_robustness(df, path):
    fams = list(dict.fromkeys(df["family"]))
    algos = list(ALGO_COLORS)
    w = 0.8 / len(algos)
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, a in enumerate(algos):
        g = df[df["algorithm"] == a].groupby("family")["ARI"].agg(["mean", "std"]).loc[fams]
        ax.bar(x + i * w, g["mean"], w, yerr=g["std"], label=a, color=ALGO_COLORS[a], capsize=2)
    ax.set_xticks(x + w * 1.5); ax.set_xticklabels(fams)
    ax.set_ylabel("ARI (середнє ± ст. відхилення)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Якість на 20 реалізаціях кожного типу (4 датчики ВЧ × 5 seed)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, path)
