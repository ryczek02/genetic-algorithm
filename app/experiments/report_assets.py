"""Generuje wykresy i tabele LaTeX do sprawozdania na podstawie wyników eksperymentów."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from functions import create_function  # noqa: E402
from gui.charts import SURFACE, build_stats_figure, build_value_figure, update_lines  # noqa: E402

from .run_experiments import CONFIG_LABELS, CONFIGS, DIMENSIONS, FUNCTIONS, REPORT  # noqa: E402

FIGURES = REPORT / "figures"
TABLES = REPORT / "generated"


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        mantissa, exponent = f"{value:.3e}".split("e")
        return f"${mantissa} \\cdot 10^{{{int(exponent)}}}$"
    return f"{value:.4f}"


def group(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["function"], r["config"], r["n_vars"])].append(r)
    return grouped


# ------------------------------------------------------------------ tabele
def config_table(function: str, grouped) -> str:
    rows = []
    best_mean = min(np.mean([r["best"] for r in grouped[(function, key, 10)]]) for key, _, _ in CONFIGS)
    for key, label, _ in CONFIGS:
        runs = grouped[(function, key, 10)]
        values = np.array([r["best"] for r in runs])
        times = np.array([r["time"] for r in runs])
        mean = fmt(values.mean())
        if np.isclose(values.mean(), best_mean):
            mean = f"\\textbf{{{mean}}}"
        rows.append(
            f"{label} & {mean} & {fmt(values.min())} & {fmt(values.max())} & {fmt(values.std())} & {times.mean():.2f} \\\\"
        )
    return "\n".join(rows)


def dimension_table(grouped) -> str:
    rows = []
    for function, (name, *_rest) in FUNCTIONS.items():
        for n_vars in DIMENSIONS:
            runs = grouped[(function, "base", n_vars)]
            values = np.array([r["best"] for r in runs])
            times = np.array([r["time"] for r in runs])
            rows.append(
                f"{name} & {n_vars} & {fmt(values.mean())} & {fmt(values.min())} & {fmt(values.max())} & {times.mean():.2f} \\\\"
            )
        rows.append("\\midrule")
    return "\n".join(rows[:-1])


def optimum_table() -> str:
    rows = []
    for function, (name, *_rest) in FUNCTIONS.items():
        for n_vars in DIMENSIONS:
            f = create_function(function, n_vars)
            point = f.optimum_point
            shown = ", ".join(f"{v:.4f}" for v in point[:3])
            rows.append(f"{name} & {n_vars} & {fmt(f.optimum_value)} & $({shown}, \\ldots)$ \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1])


def summary_table(results) -> str:
    rows = []
    by_fn_dim = defaultdict(list)
    for r in results:
        by_fn_dim[(r["function"], r["n_vars"])].append(r)
    for function, (name, *_rest) in FUNCTIONS.items():
        for n_vars in DIMENSIONS:
            best = min(by_fn_dim[(function, n_vars)], key=lambda r: r["best"])
            optimum = create_function(function, n_vars).optimum_value
            rows.append(
                f"{name} & {n_vars} & {fmt(optimum)} & {fmt(best['best'])} & {fmt(best['best'] - optimum)} & "
                f"{CONFIG_LABELS[best['config']]} \\\\"
            )
        rows.append("\\midrule")
    return "\n".join(rows[:-1])


# ----------------------------------------------------------------- wykresy
def plot_best_run(function: str, run: dict) -> None:
    h = run["history"]
    epochs = np.arange(1, len(h) + 1)
    for kind, builder, series, size in (
        ("value", build_value_figure, [(epochs, h[:, 0]), (epochs, h[:, 3])], (7.2, 3.6)),
        ("stats", build_stats_figure, [(epochs, h[:, 1]), (epochs, h[:, 2])], (7.2, 5.0)),
    ):
        fig = Figure(figsize=size, dpi=100, facecolor=SURFACE, layout="constrained")
        axes, lines = builder(fig)
        update_lines(axes, lines, series, log_scale=True)
        fig.savefig(FIGURES / f"best_{function}_{kind}.png", dpi=200, facecolor=SURFACE)


def _slice_function(function: str):
    """Funkcja 2D do wizualizacji. Dla CEC: przekrój 10-wymiarowej funkcji przez optimum (x1, x2)."""
    if function.startswith("cec"):
        f = create_function(function, 10)
        center = f.optimum_point.copy()

        def g(a, b):
            x = center.copy()
            x[0], x[1] = a, b
            return f.evaluate(x)

        return g, center[:2]
    f = create_function(function, 2)
    return (lambda a, b: f.evaluate(np.array([a, b]))), f.optimum_point


def plot_function(function: str) -> None:
    name, lower, upper = FUNCTIONS[function]
    g, opt = _slice_function(function)
    grid = np.linspace(lower, upper, 160)
    X, Y = np.meshgrid(grid, grid)
    Z = np.vectorize(g)(X, Y)
    if function == "cec2014_f4":
        Z = np.log10(Z)
    fig = plt.figure(figsize=(9.6, 4.0), facecolor=SURFACE)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, rcount=120, ccount=120)
    ax3d.set_xlabel("$x_1$")
    ax3d.set_ylabel("$x_2$")
    ax3d.set_zlabel("$\\log_{10} f$" if function == "cec2014_f4" else "$f$")
    ax3d.set_facecolor(SURFACE)
    ax = fig.add_subplot(1, 2, 2)
    mesh = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
    ax.plot(*opt[:2], marker="x", color="white", markersize=10, mew=2)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIGURES / f"function_{function}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_config_comparison(grouped) -> None:
    """Średni najlepszy wynik każdej konfiguracji, osobny panel dla każdej funkcji."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURFACE, layout="constrained")
    labels = [label for _, label, _ in CONFIGS]
    for ax, (function, (name, *_rest)) in zip(axes.flat, FUNCTIONS.items()):
        optimum = create_function(function, 10).optimum_value
        errors = [np.mean([r["best"] - optimum for r in grouped[(function, key, 10)]]) for key, _, _ in CONFIGS]
        ax.barh(labels, errors, color="#2a78d6", height=0.6)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_title(f"{name} (10 zmiennych)", loc="left", fontsize=11)
        ax.set_xlabel("średni błąd $f - f^*$ (skala log)", fontsize=9)
        ax.tick_params(labelsize=8, length=0)
        ax.grid(True, axis="x", color="#e4e3df")
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(FIGURES / "config_comparison.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def build_assets(results: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    grouped = group(results)

    for function in FUNCTIONS:
        (TABLES / f"configs_{function}.tex").write_text(config_table(function, grouped), encoding="utf-8")
        ten = [r for r in results if r["function"] == function and r["n_vars"] == 10]
        plot_best_run(function, min(ten, key=lambda r: r["best"]))
        plot_function(function)
        print("wykresy:", function)
    (TABLES / "dimensions.tex").write_text(dimension_table(grouped), encoding="utf-8")
    (TABLES / "optima.tex").write_text(optimum_table(), encoding="utf-8")
    (TABLES / "summary.tex").write_text(summary_table(results), encoding="utf-8")
    plot_config_comparison(grouped)
    print("gotowe")
