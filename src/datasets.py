"""Генератори наборів числових даних різних типів (за формою майбутніх кластерів).

Кожен генератор повертає Dataset: X (n, d), y_true (n,) -- "істинні" мітки
(використовуються лише для оцінювання, алгоритмам не передаються) та метадані.
Усі точки кластера гарантовано лежать усередині відповідної фігури
(коло/куля, еліпс, квадрат/куб, прямокутник/паралелепіпед).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler

GENERATORS = {
    "PCG64": np.random.PCG64,
    "MT19937": np.random.MT19937,
    "Philox": np.random.Philox,
    "SFC64": np.random.SFC64,
}


def make_rng(kind: str, seed: int) -> np.random.Generator:
    return np.random.Generator(GENERATORS[kind](seed))


@dataclass
class Dataset:
    key: str
    title: str
    shape: str
    orientation: str
    generation: str
    X: np.ndarray
    y: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(np.unique(self.y))


def rotation_2d(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def uniform_in_ball(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Рівномірний розподіл у d-вимірній одиничній кулі."""
    v = rng.normal(size=(n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.uniform(0, 1, n) ** (1.0 / d)
    return v * r[:, None]


def vogel_disk(n: int) -> np.ndarray:
    """Детерміноване майже рівномірне заповнення одиничного круга (спіраль Фогеля)."""
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    theta = i * np.pi * (3 - np.sqrt(5))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _assemble(parts):
    X = np.concatenate([p for p, _ in parts])
    y = np.concatenate([np.full(len(p), c) for p, c in parts])
    return X, y


# ----------------------------------------------------------------- кола / кулі
def circles_deterministic() -> Dataset:
    centers = [(-5.0, -3.0), (5.0, -3.0), (0.0, 5.0)]
    radii = [1.5, 2.0, 2.5]
    counts = [120, 160, 200]
    parts = [(np.array(c) + r * vogel_disk(n), i) for i, (c, r, n) in enumerate(zip(centers, radii, counts))]
    X, y = _assemble(parts)
    return Dataset("circles_det", "Кола різного радіуса (детермінований)", "коло", "—",
                   "детермінований (спіраль Фогеля)", X, y)


def circles_stochastic(seed: int = 0, gen: str = "MT19937", n_clusters: int = 4) -> Dataset:
    rng = make_rng(gen, seed)
    centers, radii = [], []
    while len(centers) < n_clusters:
        c, r = rng.uniform(-8, 8, 2), rng.uniform(1.0, 2.2)
        if all(np.linalg.norm(c - c2) > r + r2 + 0.6 for c2, r2 in zip(centers, radii)):
            centers.append(c)
            radii.append(r)
    parts = [(c + r * uniform_in_ball(int(rng.integers(100, 201)), 2, rng), i)
             for i, (c, r) in enumerate(zip(centers, radii))]
    X, y = _assemble(parts)
    return Dataset("circles_stoch", "Кола випадкового радіуса (стохастичний)", "коло", "—",
                   f"стохастичний ({gen}, рівномірний у крузі)", X, y, {"seed": seed})


def balls_3d(seed: int = 5, gen: str = "PCG64") -> Dataset:
    rng = make_rng(gen, seed)
    centers = np.array([[0, 0, 0], [6, 0, 0], [0, 6, 0], [0, 0, 6]], dtype=float)
    parts = [(c + rng.uniform(1.2, 2.2) * uniform_in_ball(150, 3, rng), i) for i, c in enumerate(centers)]
    X, y = _assemble(parts)
    return Dataset("balls_3d", "Кулі у тривимірному просторі", "куля (3D)", "—",
                   f"стохастичний ({gen}, рівномірний у кулі)", X, y)


# ----------------------------------------------------------------- еліпси
def ellipses(parallel: bool, seed: int, gen: str = "PCG64") -> Dataset:
    rng = make_rng(gen, seed)
    centers = [(-4.0, -3.5), (4.0, -3.5), (-4.0, 3.5), (4.0, 3.5)]
    parts = []
    for i, c in enumerate(centers):
        a, b = rng.uniform(2.6, 3.4), rng.uniform(0.35, 0.6)
        theta = 0.0 if parallel else rng.uniform(0, np.pi)
        pts = uniform_in_ball(150, 2, rng) * np.array([a, b])
        parts.append((np.array(c) + pts @ rotation_2d(theta).T, i))
    X, y = _assemble(parts)
    tag = "паралельні осі" if parallel else "непаралельні осі"
    key = "ellipses_parallel" if parallel else "ellipses_nonparallel"
    return Dataset(key, f"Еліпси, {tag}", "еліпс", tag, f"стохастичний ({gen}, рівномірний в еліпсі)", X, y)


# ----------------------------------------------------------------- квадрати / прямокутники / паралелепіпеди
def squares_parallel_deterministic() -> Dataset:
    parts = []
    for i, (c, side) in enumerate(zip([(-5, -5), (5, -5), (-5, 5), (5, 5)], [3.0, 3.5, 4.0, 3.0])):
        g = np.linspace(-side / 2, side / 2, 13)
        gx, gy = np.meshgrid(g, g)
        parts.append((np.column_stack([gx.ravel(), gy.ravel()]) + np.array(c), i))
    X, y = _assemble(parts)
    return Dataset("squares_parallel", "Квадрати, паралельні сторони (детермінований)", "квадрат",
                   "паралельні сторони", "детермінований (регулярна решітка)", X, y)


def rectangles(square: bool, parallel: bool, seed: int, gen: str) -> Dataset:
    rng = make_rng(gen, seed)
    centers = [(-4.0, -3.0), (4.0, -3.0), (-4.0, 3.0), (4.0, 3.0)]
    parts = []
    for i, c in enumerate(centers):
        w = rng.uniform(1.6, 2.2)
        h = w if square else rng.uniform(0.3, 0.5)
        if not square:
            w *= 1.5
        theta = 0.0 if parallel else rng.uniform(0, np.pi)
        pts = rng.uniform(-1, 1, size=(150, 2)) * np.array([w, h])
        parts.append((np.array(c) + pts @ rotation_2d(theta).T, i))
    X, y = _assemble(parts)
    shape = "квадрат" if square else "прямокутник"
    tag = "паралельні сторони" if parallel else "непаралельні сторони"
    key = f"{'squares' if square else 'rectangles'}_{'parallel' if parallel else 'nonparallel'}"
    title = f"{'Квадрати' if square else 'Прямокутники'}, {tag}"
    return Dataset(key, title, shape, tag, f"стохастичний ({gen}, рівномірний у фігурі)", X, y)


def boxes_3d_nonparallel(seed: int = 7, gen: str = "SFC64") -> Dataset:
    rng = make_rng(gen, seed)
    centers = np.array([[0, 0, 0], [7, 0, 0], [0, 7, 0], [0, 0, 7]], dtype=float)
    parts = []
    for i, c in enumerate(centers):
        half = np.array([rng.uniform(2.0, 2.8), rng.uniform(0.6, 1.0), rng.uniform(0.3, 0.6)])
        pts = rng.uniform(-1, 1, size=(150, 3)) * half
        parts.append((c + pts @ random_rotation(3, rng).T, i))
    X, y = _assemble(parts)
    return Dataset("boxes_3d", "Паралелепіпеди у 3D, непаралельні грані", "паралелепіпед (3D)",
                   "непаралельні грані", f"стохастичний ({gen}, рівномірний у фігурі)", X, y)


# ----------------------------------------------------------------- несиметричні фігури (літери)
LETTER_STROKES = {
    "Г": [[(0, 0), (0, 2), (1.2, 2)]],
    "С": [[(1.2, 2), (0, 2), (0, 0), (1.2, 0)]],
    "П": [[(0, 0), (0, 2), (1.2, 2), (1.2, 0)]],
    "Т": [[(0, 2), (1.2, 2)], [(0.6, 2), (0.6, 0)]],
    "Е": [[(1.2, 2), (0, 2), (0, 0), (1.2, 0)], [(0, 1), (1.0, 1)]],
    "Х": [[(0, 0), (1.2, 2)], [(0, 2), (1.2, 0)]],
}


def sample_polyline(points, n, width, rng):
    """Точки вздовж ламаної з гаусовим шумом упоперек лінії."""
    pts = np.asarray(points, dtype=float)
    seg = np.diff(pts, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(lens)])
    s = rng.uniform(0, cum[-1], n)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(lens) - 1)
    t = (s - cum[idx]) / lens[idx]
    base = pts[idx] + seg[idx] * t[:, None]
    direction = seg[idx] / lens[idx, None]
    normal = np.column_stack([-direction[:, 1], direction[:, 0]])
    return base + normal * rng.normal(scale=width, size=n)[:, None]


def letters(letters_seq, seed: int, gen: str, spacing: float, width: float, key: str) -> Dataset:
    rng = make_rng(gen, seed)
    parts = []
    for i, letter in enumerate(letters_seq):
        strokes = LETTER_STROKES[letter]
        lens = np.array([np.linalg.norm(np.diff(np.asarray(s, float), axis=0), axis=1).sum() for s in strokes])
        counts = np.round(200 * lens / lens.sum()).astype(int)
        pts = np.concatenate([sample_polyline(s, k, width, rng) for s, k in zip(strokes, counts)])
        pts[:, 0] += i * spacing
        parts.append((pts, i))
    X, y = _assemble(parts)
    return Dataset(key, f"Літери {', '.join(letters_seq)}", "несиметрична (літера)", "—",
                   f"стохастичний ({gen}, нормальний шум уздовж штрихів)", X, y)


def rings(seed: int = 9, gen: str = "MT19937") -> Dataset:
    """Інший варіант (п.2.7): концентричні кільця + ядро — класичний неопуклий випадок."""
    rng = make_rng(gen, seed)
    parts = [(1.0 * uniform_in_ball(150, 2, rng), 0)]
    for i, (r, n) in enumerate([(3.0, 250), (5.5, 350)], start=1):
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = r + rng.normal(scale=0.2, size=n)
        parts.append((np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]), i))
    X, y = _assemble(parts)
    return Dataset("rings", "Концентричні кільця з ядром", "кільце", "—",
                   f"стохастичний ({gen}, нормальний шум по радіусу)", X, y)


def mixture(seed: int = 4, gen: str = "Philox") -> Dataset:
    """Суміш типів: коло, повернутий еліпс, квадрат, повернутий прямокутник, літера С."""
    rng = make_rng(gen, seed)
    parts = [((-6, -6) + 1.4 * uniform_in_ball(150, 2, rng), 0)]
    ell = uniform_in_ball(150, 2, rng) * np.array([2.2, 0.6]) @ rotation_2d(rng.uniform(0, np.pi)).T
    parts.append(((6, -6) + ell, 1))
    parts.append(((-6, 6) + rng.uniform(-1.5, 1.5, size=(150, 2)), 2))
    rect = rng.uniform(-1, 1, size=(150, 2)) * np.array([2.2, 0.6]) @ rotation_2d(rng.uniform(0, np.pi)).T
    parts.append(((6, 6) + rect, 3))
    c_letter = sample_polyline(LETTER_STROKES["С"][0], 150, 0.1, rng) * 1.5 + np.array([-0.9, -1.5])
    parts.append((c_letter, 4))
    X, y = _assemble(parts)
    return Dataset("mixture", "Суміш типів (коло, еліпс, квадрат, прямокутник, літера)", "суміш",
                   "різна", f"стохастичний ({gen})", X, y)


# ----------------------------------------------------------------- реальні дані
def real_iris() -> Dataset:
    d = load_iris()
    return Dataset("real_iris", "Iris (реальні дані, 4 ознаки)", "реальні дані", "—",
                   "реальний набір (Fisher, 1936)", d.data, d.target)


def real_wine() -> Dataset:
    d = load_wine()
    X = StandardScaler().fit_transform(d.data)
    return Dataset("real_wine", "Wine (реальні дані, 13 ознак, стандартизовано)", "реальні дані", "—",
                   "реальний набір (UCI Wine)", X, d.target)


def all_datasets() -> dict[str, Dataset]:
    items = [
        circles_deterministic(),
        circles_stochastic(),
        balls_3d(),
        ellipses(parallel=True, seed=10),
        ellipses(parallel=False, seed=11),
        squares_parallel_deterministic(),
        rectangles(square=True, parallel=False, seed=20, gen="Philox"),
        rectangles(square=False, parallel=True, seed=21, gen="SFC64"),
        rectangles(square=False, parallel=False, seed=22, gen="SFC64"),
        boxes_3d_nonparallel(),
        letters(("Г", "С", "Т"), seed=3, gen="PCG64", spacing=2.2, width=0.08, key="letters_gst"),
        letters(("П", "Е", "Х"), seed=6, gen="MT19937", spacing=1.9, width=0.1, key="letters_pex"),
        rings(),
        mixture(),
        real_iris(),
        real_wine(),
    ]
    return {d.key: d for d in items}


# Фабрики для дослідження впливу датчика випадкових чисел (різні seed і генератори)
FAMILIES = {
    "circles_stoch": lambda seed, gen: circles_stochastic(seed=seed, gen=gen),
    "ellipses_nonparallel": lambda seed, gen: ellipses(parallel=False, seed=seed, gen=gen),
    "rectangles_nonparallel": lambda seed, gen: rectangles(square=False, parallel=False, seed=seed, gen=gen),
    "letters_gst": lambda seed, gen: letters(("Г", "С", "Т"), seed=seed, gen=gen, spacing=2.2, width=0.08,
                                             key="letters_gst"),
}
