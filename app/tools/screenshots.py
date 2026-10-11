"""Automatycznie wykonuje zrzuty ekranu aplikacji dla sprawozdania (tylko macOS).

Uruchomienie: python -m tools.screenshots
"""
from __future__ import annotations

import os
import subprocess
import warnings
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import Quartz  # noqa: E402

from ga import GAConfig  # noqa: E402
from gui import GeneticAlgorithmApp  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "report" / "figures" / "screens"

BASE = GAConfig(population_size=100, epochs=500, seed=7)
SCENARIOS = [
    ("01_start", replace(BASE), None, 0),
    ("02_ackley", replace(BASE, function="ackley", lower=-32.768, upper=32.768), "run", 0),
    ("03_schwefel", replace(BASE, function="schwefel", lower=-500, upper=500, selection="roulette",
                            crossover="uniform", epochs=800), "run", 1),
    ("04_cec_f4", replace(BASE, function="cec2014_f4", lower=-100, upper=100, epochs=1000,
                          crossover="granular", grain_size=8, mutation="two_point"), "run_log", 0),
    ("05_cec_f8", replace(BASE, function="cec2014_f8", lower=-100, upper=100, epochs=1000,
                          selection="best", selection_percentage=0.3, mutation="boundary"), "run", 1),
    ("06_maks", replace(BASE, function="ackley", lower=-32.768, upper=32.768, maximize=True, n_vars=5,
                        epochs=300, elite_mode="count", elite_value=2), "run", 0),
]


def window_id() -> int:
    windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    own = [w for w in windows if w.get("kCGWindowOwnerPID") == os.getpid()]
    if not own:
        raise RuntimeError("Nie znaleziono okna aplikacji")
    size = lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"]
    return int(max(own, key=size)["kCGWindowNumber"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    app = GeneticAlgorithmApp()
    app.lift()
    app.attributes("-topmost", True)
    steps = iter(SCENARIOS)

    def capture(name: str, then) -> None:
        app.update_idletasks()
        subprocess.run(["screencapture", "-x", "-o", f"-l{window_id()}", str(OUT / f"{name}.png")], check=True)
        print("zapisano", name)
        app.after(300, then)

    def wait_for_run(name: str) -> None:
        if app._worker and app._worker.is_alive():
            app.after(200, lambda: wait_for_run(name))
        else:
            app.after(900, lambda: capture(name, next_step))

    def next_step() -> None:
        try:
            name, cfg, action, tab = next(steps)
        except StopIteration:
            app.destroy()
            return
        app.focus_force()
        app.form.set_config(cfg)
        app.charts.select(tab)
        app.log_var.set(action == "run_log")
        app._toggle_log()
        if action is None:
            app.after(1200, lambda: capture(name, next_step))
        else:
            app.start()
            app.after(500, lambda: wait_for_run(name))

    app.after(1500, next_step)
    app.mainloop()


if __name__ == "__main__":
    main()
