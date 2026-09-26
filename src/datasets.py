"""Генератори наборів числових даних різних типів (форм майбутніх кластерів).

Кожна функція повертає кортеж (X, y_true, opis), де:
    X       -- np.ndarray форми (n, 2) або (n, d) з координатами точок;
    y_true  -- np.ndarray форми (n,) з номером "істинного" кластера точки
               (потрібен лише для оцінки якості та не подається алгоритмам);
    opis    -- рядок з коротким описом набору (для звіту / підписів графіків).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris


def _rng(seed):
    return np.random.default_rng(seed)


def circles_deterministic(n_clusters: int = 3, n_points: int = 120, radius: float = 1.5) -> tuple:
    """Кластери-кола: центри розташовані детерміновано (правильний багатокутник),
    точки в межах кожного кола -- за регулярною полярною сіткою (без випадковості)."""
    xs, ys, labels = [], [], []
    layout_r = 5.0
    for c in range(n_clusters):
        angle = 2 * np.pi * c / n_clusters
        cx, cy = layout_r * np.cos(angle), layout_r * np.sin(angle)
        k = int(np.ceil(np.sqrt(n_points)))
        rr = np.linspace(0.15, 1.0, k)
        th = np.linspace(0, 2 * np.pi, k, endpoint=False)
        RR, TH = np.meshgrid(rr, th)
        RR, TH = RR.ravel()[:n_points], TH.ravel()[:n_points]
        xs.append(cx + radius * RR * np.cos(TH))
        ys.append(cy + radius * RR * np.sin(TH))
        labels.append(np.full(len(RR), c))
    X = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    y = np.concatenate(labels)
    return X, y, f"Кола, {n_clusters} кластери, детермінований (регулярна сітка) розподіл"


def circles_stochastic(n_clusters: int = 3, n_points: int = 150, seed: int = 0) -> tuple:
    """Кластери-кола: центри та радіуси випадкові (uniform), точки всередині
    кола -- за допомогою нормального (Gauss) генератора, обрізаного по радіусу."""
    rng = _rng(seed)
    xs, ys, labels = [], [], []
    for c in range(n_clusters):
        cx, cy = rng.uniform(-8, 8, size=2)
        radius = rng.uniform(0.8, 2.2)
        pts = rng.normal(scale=radius / 2.2, size=(n_points, 2))
        norm = np.linalg.norm(pts, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-9, None)
        scale = np.minimum(1.0, radius / norm)
        pts = pts  # normal already gives roughly circular disk after clipping outliers
        pts = np.clip(pts, -radius, radius)
        xs.append(cx + pts[:, 0])
        ys.append(cy + pts[:, 1])
        labels.append(np.full(n_points, c))
    X = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    y = np.concatenate(labels)
    return X, y, f"Кола, {n_clusters} кластери, стохастичний (нормальний розподіл) генератор, seed={seed}"


def ellipses(n_clusters: int = 4, n_points: int = 150, parallel: bool = True, seed: int = 1) -> tuple:
    """Кластери-еліпси. parallel=True -- усі осі еліпсів співнапрямлені (кут=0),
    parallel=False -- кожен еліпс повернутий на власний випадковий кут."""
    rng = _rng(seed)
    xs, ys, labels = [], [], []
    layout_r = 6.0
    for c in range(n_clusters):
        angle_pos = 2 * np.pi * c / n_clusters
        cx, cy = layout_r * np.cos(angle_pos), layout_r * np.sin(angle_pos)
        a, b = rng.uniform(1.2, 2.5), rng.uniform(0.3, 0.9)
        theta = 0.0 if parallel else rng.uniform(0, np.pi)
        t = rng.normal(size=n_points)
        u = rng.normal(size=n_points)
        pts = np.column_stack([a * t, b * u])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = pts @ rot.T
        xs.append(cx + pts[:, 0])
        ys.append(cy + pts[:, 1])
        labels.append(np.full(n_points, c))
    X = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    y = np.concatenate(labels)
    tag = "паралельні осі" if parallel else "непаралельні (випадково повернуті) осі"
    return X, y, f"Еліпси, {n_clusters} кластери, {tag}, seed={seed}"


def rectangles(n_clusters: int = 4, n_points: int = 150, square: bool = False,
               parallel: bool = True, seed: int = 2) -> tuple:
    """Кластери-прямокутники (square=True -> квадрати/куби). parallel визначає,
    чи сторони всіх фігур взаємно паралельні, чи кожна повернута на свій кут."""
    rng = _rng(seed)
    xs, ys, labels = [], [], []
    layout_r = 6.0
    for c in range(n_clusters):
        angle_pos = 2 * np.pi * c / n_clusters
        cx, cy = layout_r * np.cos(angle_pos), layout_r * np.sin(angle_pos)
        w = rng.uniform(1.0, 2.2)
        h = w if square else rng.uniform(0.4, 1.0)
        theta = 0.0 if parallel else rng.uniform(0, np.pi / 2)
        pts = rng.uniform(-1, 1, size=(n_points, 2)) * np.array([w, h])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = pts @ rot.T
        xs.append(cx + pts[:, 0])
        ys.append(cy + pts[:, 1])
        labels.append(np.full(n_points, c))
    X = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    y = np.concatenate(labels)
    shape_name = "квадрати" if square else "прямокутники"
    tag = "паралельні сторони" if parallel else "непаралельні (випадково повернуті) сторони"
    return X, y, f"{shape_name.capitalize()}, {n_clusters} кластери, {tag}, seed={seed}"


def _sample_polyline(points_xy, n_points, width, rng):
    """Рівномірно вибирає точки вздовж ламаної (список опорних точок) з гаусовим шумом
    упоперек лінії -- використовується для побудови кластерів у формі літер."""
    points_xy = np.asarray(points_xy, dtype=float)
    seglens = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    seglens = np.maximum(seglens, 1e-9)
    cum = np.concatenate([[0], np.cumsum(seglens)])
    total = cum[-1]
    s = rng.uniform(0, total, size=n_points)
    seg_idx = np.searchsorted(cum, s, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seglens) - 1)
    t = (s - cum[seg_idx]) / seglens[seg_idx]
    p0 = points_xy[seg_idx]
    p1 = points_xy[seg_idx + 1]
    base = p0 + (p1 - p0) * t[:, None]
    direction = (p1 - p0) / seglens[seg_idx, None]
    normal = np.column_stack([-direction[:, 1], direction[:, 0]])
    noise = rng.normal(scale=width, size=n_points)
    return base + normal * noise[:, None]


_LETTER_STROKES = {
    "Г": [[(0, 0), (0, 2), (1.2, 2)]],
    "С": [[(1.2, 2), (0, 2), (0, 0), (1.2, 0)]],
    "П": [[(0, 0), (0, 2), (1.2, 2), (1.2, 0)]],
    "Т": [[(0, 2), (1.2, 2)], [(0.6, 2), (0.6, 0)]],
    "Е": [[(1.2, 2), (0, 2), (0, 0), (1.2, 0)], [(0, 1), (1.0, 1)]],
    "Х": [[(0, 0), (1.2, 2)], [(0, 2), (1.2, 0)]],
}


def letter_clusters(letters=("Г", "С", "Т"), n_points: int = 200, width: float = 0.08,
                     seed: int = 3) -> tuple:
    """Несиметричні кластери у формі літер (Г, С, П, Т, Е, Х)."""
    rng = _rng(seed)
    xs, ys, labels = [], [], []
    spacing = 2.2
    for idx, letter in enumerate(letters):
        strokes = _LETTER_STROKES[letter]
        per_stroke = max(1, n_points // len(strokes))
        pts_all = []
        for stroke in strokes:
            pts_all.append(_sample_polyline(stroke, per_stroke, width, rng))
        pts = np.concatenate(pts_all, axis=0)
        pts[:, 0] += idx * spacing
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
        labels.append(np.full(len(pts), idx))
    X = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    y = np.concatenate(labels)
    return X, y, f"Несиметричні кластери у формі літер {', '.join(letters)}, seed={seed}"


def mixture(n_points: int = 150, seed: int = 4) -> tuple:
    """Суміш різних типів кластерів у одному наборі: коло + еліпс (повернутий)
    + квадрат + прямокутник (повернутий)."""
    rng = _rng(seed)
    parts = []
    labels = []

    cx, cy, r = -6, -6, 1.4
    ang = rng.uniform(0, 2 * np.pi, n_points)
    rad = r * np.sqrt(rng.uniform(0, 1, n_points))
    parts.append(np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)]))
    labels.append(np.full(n_points, 0))

    cx, cy = 6, -6
    a, b, theta = 2.0, 0.6, rng.uniform(0, np.pi)
    t, u = rng.normal(size=n_points), rng.normal(size=n_points)
    pts = np.column_stack([a * t, b * u])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ rot.T
    parts.append(np.column_stack([cx + pts[:, 0], cy + pts[:, 1]]))
    labels.append(np.full(n_points, 1))

    cx, cy, w = -6, 6, 1.6
    pts = rng.uniform(-w, w, size=(n_points, 2))
    parts.append(np.column_stack([cx + pts[:, 0], cy + pts[:, 1]]))
    labels.append(np.full(n_points, 2))

    cx, cy, w, h, theta = 6, 6, 2.0, 0.7, rng.uniform(0, np.pi / 2)
    pts = rng.uniform(-1, 1, size=(n_points, 2)) * np.array([w, h])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ rot.T
    parts.append(np.column_stack([cx + pts[:, 0], cy + pts[:, 1]]))
    labels.append(np.full(n_points, 3))

    X = np.concatenate(parts, axis=0)
    y = np.concatenate(labels)
    return X, y, "Суміш типів: коло, повернутий еліпс, квадрат, повернутий прямокутник"


def real_dataset() -> tuple:
    """Реальний набір даних Iris (Fisher, 1936) -- 4 ознаки, 3 класи квітів."""
    data = load_iris()
    return data.data, data.target, "Реальний набір даних Iris (4 ознаки, 3 класи)"


def all_datasets() -> dict:
    """Повертає впорядкований словник {назва: (X, y_true, опис)} з усіма наборами,
    що використовуються в дослідженні."""
    return {
        "circles_det": circles_deterministic(),
        "circles_stoch": circles_stochastic(),
        "ellipses_parallel": ellipses(parallel=True, seed=10),
        "ellipses_nonparallel": ellipses(parallel=False, seed=11),
        "squares_nonparallel": rectangles(square=True, parallel=False, seed=20),
        "rectangles_parallel": rectangles(square=False, parallel=True, seed=21),
        "letters": letter_clusters(),
        "mixture": mixture(),
        "real_iris": real_dataset(),
    }
