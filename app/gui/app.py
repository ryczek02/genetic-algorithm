from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font, messagebox, ttk

from functions import create_function
from ga import GAConfig, build_algorithm
from ga.algorithm import EpochStats, RunResult
from storage import CsvResultRepository

from .charts import ChartPanel
from .form import ParameterForm

AUTHORS = "Wykonali: Łukasz Ryczko, Maciej Malina"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHART_REFRESH_S = 0.25


class StatCard(ttk.LabelFrame):
    def __init__(self, master, title: str, value_font):
        super().__init__(master, text=title, padding=(10, 2, 10, 6))
        self.value = ttk.Label(self, text="-", font=value_font)
        self.value.pack(anchor="w")

    def set(self, text: str) -> None:
        self.value.config(text=text)


class GeneticAlgorithmApp(tk.Tk):
    """Główne okno: formularz parametrów po lewej, wyniki i wykresy po prawej."""

    def __init__(self, repository: CsvResultRepository | None = None):
        super().__init__()
        self.title("Algorytm genetyczny | Projekt 1 | Obliczenia ewolucyjne")
        self.geometry("1460x860")
        self.minsize(1300, 800)
        self.repository = repository or CsvResultRepository(RESULTS_DIR)

        self._events: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._history: list[EpochStats] = []
        self._started_at = 0.0
        self._last_chart = 0.0
        self._optimum: float | None = None
        self._config: GAConfig | None = None
        self.last_run_id: str | None = None

        self._init_fonts()
        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ wygląd
    def _init_fonts(self) -> None:
        base = font.nametofont("TkDefaultFont")
        size = base.cget("size")
        self.font_title = base.copy()
        self.font_title.configure(size=size + 9, weight="bold")
        self.font_value = base.copy()
        self.font_value.configure(size=size + 3, weight="bold")
        hint = base.copy()
        hint.configure(size=max(size - 2, 9))
        style = ttk.Style(self)
        style.configure("Hint.TLabel", font=hint, foreground="#6b6a66")
        style.configure("Subtitle.TLabel", foreground="#6b6a66")

    def _build(self) -> None:
        root = ttk.Frame(self, padding=(18, 14, 18, 10))
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(header, text="Klasyczny algorytm genetyczny", font=self.font_title).pack(anchor="w")

        left = ttk.Frame(root)
        left.grid(row=1, column=0, sticky="ns", padx=(0, 18))
        self.form = ParameterForm(left)
        self.form.pack(fill="x")
        buttons = self.form.action_area()
        self.start_button = ttk.Button(buttons, text="Start", command=self.start, default="active")
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Przywróć domyślne", command=lambda: self.form.set_config(GAConfig())).pack(
            side="right"
        )
        self.bind("<Return>", lambda _e: self.start())

        right = ttk.Frame(root)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        cards = ttk.Frame(right)
        cards.grid(row=0, column=0, sticky="ew")
        self.cards = {}
        for i, (key, title) in enumerate(
            (("time", "Czas obliczeń"), ("best", "Najlepsze f(x)"), ("epoch", "Epoka"), ("error", "Błąd |f - f*|"))
        ):
            cards.columnconfigure(i, weight=1, uniform="card")
            card = StatCard(cards, title, self.font_value)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            self.cards[key] = card

        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(10, 8))

        best_box = ttk.LabelFrame(right, text="Najlepszy osobnik x", padding=(8, 4))
        best_box.grid(row=2, column=0, sticky="ew")
        self.best_x = tk.Text(best_box, width=40, height=3, wrap="word", relief="flat", borderwidth=0, highlightthickness=0)
        self.best_x.configure(font=font.nametofont("TkFixedFont"), state="disabled")
        self.best_x.pack(fill="x")

        tools = ttk.Frame(right)
        tools.grid(row=3, column=0, sticky="ew", pady=(10, 4))
        self.log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tools, text="Skala logarytmiczna", variable=self.log_var, command=self._toggle_log).pack(
            side="left"
        )
        ttk.Button(tools, text="Otwórz folder wyników", command=self._open_results).pack(side="right")
        ttk.Button(tools, text="Zapisz wykresy PNG", command=self.save_charts).pack(side="right", padx=6)

        self.charts = ChartPanel(right)
        self.charts.grid(row=4, column=0, sticky="nsew")

        footer = ttk.Frame(root)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.status = ttk.Label(footer, text="Gotowy. Ustaw parametry i kliknij Start.", style="Subtitle.TLabel")
        self.status.pack(side="left")
        ttk.Label(footer, text=AUTHORS, style="Subtitle.TLabel").pack(side="right")

    # ------------------------------------------------------------- uruchamianie
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        try:
            config = self.form.get_config()
            algorithm = build_algorithm(config)
        except ValueError as error:
            messagebox.showerror("Niepoprawne parametry", str(error), parent=self)
            return

        self._config = config
        self._optimum = None if config.maximize else create_function(config.function, config.n_vars).optimum_value
        self._history = []
        self._stop.clear()
        self.charts.clear()
        for card in self.cards.values():
            card.set("-")
        self._set_best_x("")
        self.progress.configure(maximum=config.epochs, value=0)
        self.form.set_running(True)
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.status.config(
            text=f"Obliczenia: {algorithm.objective.name}, {config.n_vars} zmiennych, "
            f"{algorithm.encoder.bits_per_var} bitów na zmienną..."
        )
        self._started_at = time.perf_counter()
        self._worker = threading.Thread(target=self._run_worker, args=(algorithm,), daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_worker(self, algorithm) -> None:
        try:
            result = algorithm.run(on_epoch=lambda s: self._events.put(("epoch", s)), should_stop=self._stop.is_set)
            run_id = self.repository.save_run(result)
            self._events.put(("done", result, run_id))
        except Exception as error:  # błąd biblioteki funkcji nie może zawiesić GUI
            self._events.put(("error", error))

    # --------------------------------------------------------- odświeżanie GUI
    def _poll(self) -> None:
        latest = None
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == "epoch":
                    self._history.append(event[1])
                    latest = event[1]
                elif event[0] == "done":
                    self._finish(*event[1:])
                elif event[0] == "error":
                    self._fail(event[1])
        except queue.Empty:
            pass

        if self._worker and self._worker.is_alive():
            self.cards["time"].set(f"{time.perf_counter() - self._started_at:.2f} s")
        if latest is not None:
            self._show_stats(latest)
            now = time.perf_counter()
            if now - self._last_chart >= CHART_REFRESH_S:
                self.charts.show(self._history)
                self._last_chart = now
        self.after(50, self._poll)

    def _show_stats(self, stats: EpochStats) -> None:
        self.cards["best"].set(f"{stats.best_so_far:.6g}")
        self.cards["epoch"].set(f"{stats.epoch} / {self._config.epochs}")
        if self._optimum is not None:
            self.cards["error"].set(f"{abs(stats.best_so_far - self._optimum):.4g}")
        self.progress.configure(value=stats.epoch)

    def _finish(self, result: RunResult, run_id: str) -> None:
        self.last_run_id = run_id
        if self._history:
            self._show_stats(self._history[-1])
        self.charts.show(self._history)
        self.cards["time"].set(f"{result.elapsed:.2f} s")
        self._set_best_x(", ".join(f"{v:.6f}" for v in result.best_x))
        state = "Zatrzymano" if result.stopped else "Zakończono"
        self.status.config(
            text=f"{state} po {len(result.history)} epokach w {result.elapsed:.2f} s. "
            f"Wyniki zapisano w results/run_{run_id}.csv"
        )
        self._reset_controls()

    def _fail(self, error: Exception) -> None:
        self._reset_controls()
        self.status.config(text="Błąd obliczeń.")
        messagebox.showerror("Błąd", str(error), parent=self)

    def _reset_controls(self) -> None:
        self.form.set_running(False)
        self.start_button.state(["!disabled"])
        self.stop_button.state(["disabled"])

    def _set_best_x(self, text: str) -> None:
        self.best_x.configure(state="normal")
        self.best_x.delete("1.0", "end")
        self.best_x.insert("1.0", text)
        self.best_x.configure(state="disabled")

    # -------------------------------------------------------------- narzędzia
    def _toggle_log(self) -> None:
        self.charts.log_scale = self.log_var.get()
        self.charts.show(self._history)

    def save_charts(self) -> None:
        if not self._history:
            messagebox.showinfo("Brak danych", "Najpierw uruchom algorytm.", parent=self)
            return
        prefix = f"run_{self.last_run_id}" if self.last_run_id else "run"
        paths = self.charts.save(RESULTS_DIR / "plots", prefix)
        self.status.config(text="Zapisano wykresy: " + ", ".join(p.name for p in paths))

    def _open_results(self) -> None:
        RESULTS_DIR.mkdir(exist_ok=True)
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(RESULTS_DIR)])
