from __future__ import annotations

from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ga.algorithm import EpochStats  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def build_value_figure(figure: Figure):
    ax = figure.add_subplot(111)
    style_axes(ax, "Wartość funkcji celu w kolejnych epokach", "f(x)")
    ax.set_xlabel("Epoka", color=TEXT_MUTED, fontsize=9)
    # "najlepszy dotychczas" rysowany pod spodem i szerzej: przy elitaryzmie obie linie się pokrywają
    so_far, = ax.plot([], [], color=SERIES_2, linewidth=4.0, alpha=0.55, label="Najlepszy dotychczas", zorder=2)
    best, = ax.plot([], [], color=SERIES_1, linewidth=1.4, label="Najlepszy w epoce", zorder=3)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    return [ax], [best, so_far]


def build_stats_figure(figure: Figure):
    top = figure.add_subplot(211)
    bottom = figure.add_subplot(212, sharex=top)
    style_axes(top, "Średnia wartość funkcji w populacji", "średnia f(x)")
    style_axes(bottom, "Odchylenie standardowe w populacji", "odchylenie std")
    bottom.set_xlabel("Epoka", color=TEXT_MUTED, fontsize=9)
    mean, = top.plot([], [], color=SERIES_1, linewidth=1.4)
    std, = bottom.plot([], [], color=SERIES_2, linewidth=1.0)
    return [top, bottom], [mean, std]


def update_lines(axes, lines, series, log_scale: bool) -> None:
    for line, (xs, ys) in zip(lines, series):
        line.set_data(xs, ys)
    for ax in axes:
        values = [y for line in lines if line.axes is ax for y in line.get_ydata()]
        if log_scale and values:
            ax.set_yscale("log" if min(values) > 0 else "symlog")
        else:
            ax.set_yscale("linear")
        ax.relim()
        ax.autoscale_view()


def history_series(history: list[EpochStats]):
    epochs = [s.epoch for s in history]
    return {
        "value": [(epochs, [s.best for s in history]), (epochs, [s.best_so_far for s in history])],
        "stats": [(epochs, [s.mean for s in history]), (epochs, [s.std for s in history])],
    }


class ChartPanel(ttk.Notebook):
    """Dwie zakładki z wykresami odświeżanymi na żywo w trakcie obliczeń."""

    def __init__(self, master):
        super().__init__(master)
        self.log_scale = False
        self._charts = {}
        for key, title, builder in (
            ("value", "Wartość funkcji", build_value_figure),
            ("stats", "Średnia i odchylenie std", build_stats_figure),
        ):
            frame = ttk.Frame(self)
            figure = Figure(figsize=(6.4, 4.6), dpi=100, facecolor=SURFACE, layout="constrained")
            axes, lines = builder(figure)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.add(frame, text=title)
            self._charts[key] = (figure, axes, lines, canvas)

    def show(self, history: list[EpochStats]) -> None:
        series = history_series(history)
        for key, (_, axes, lines, canvas) in self._charts.items():
            update_lines(axes, lines, series[key], self.log_scale)
            canvas.draw_idle()

    def clear(self) -> None:
        self.show([])

    def save(self, directory: Path, prefix: str) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for key, (figure, *_rest) in self._charts.items():
            path = directory / f"{prefix}_{key}.png"
            figure.savefig(path, dpi=200, facecolor=SURFACE)
            paths.append(path)
        return paths
