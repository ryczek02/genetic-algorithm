"""Eksperymenty do sprawozdania: każda konfiguracja uruchamiana 10 razy.

Uruchomienie (z katalogu app/):
    python -m experiments.run_experiments            # obliczenia + wykresy + tabele
    python -m experiments.run_experiments --plots    # tylko wykresy i tabele z zapisanych wyników
"""
from __future__ import annotations

import argparse
import csv
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from ga import GAConfig, build_algorithm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
CACHE = Path(__file__).resolve().parents[1] / "results" / "experiments.pkl"
RUNS = 10

FUNCTIONS = {
    "ackley": ("Ackley", -32.768, 32.768),
    "schwefel": ("Schwefel", -500.0, 500.0),
    "cec2014_f4": ("CEC2014 F4", -100.0, 100.0),
    "cec2014_f8": ("CEC2014 F8", -100.0, 100.0),
}

BASE = GAConfig(
    population_size=100, epochs=1000, precision=1e-4,
    selection="tournament", tournament_size=3, selection_percentage=0.3,
    crossover="two_point", crossover_prob=0.8, grain_size=8,
    mutation="one_point", mutation_prob=0.2, inversion_prob=0.05,
    elite_value=0.05, elite_mode="percent",
)

# (klucz, opis w tabeli, zmiany względem konfiguracji bazowej)
CONFIGS = [
    ("base", "Bazowa", {}),
    ("sel_best", "Selekcja najlepszych", {"selection": "best"}),
    ("sel_worst", "Selekcja najgorszych", {"selection": "worst"}),
    ("sel_roulette", "Koło ruletki", {"selection": "roulette"}),
    ("cx_one", "Krzyżowanie 1-punktowe", {"crossover": "one_point"}),
    ("cx_uniform", "Krzyżowanie jednorodne", {"crossover": "uniform"}),
    ("cx_granular", "Krzyżowanie ziarniste", {"crossover": "granular"}),
    ("mut_boundary", "Mutacja brzegowa", {"mutation": "boundary"}),
    ("mut_two", "Mutacja 2-punktowa", {"mutation": "two_point"}),
    ("no_elite", "Bez elitaryzmu", {"elite_value": 0.0}),
]
CONFIG_LABELS = {key: label for key, label, _ in CONFIGS}
DIMENSIONS = (10, 20, 30)


def _tasks():
    tasks = []
    for function, (_, lower, upper) in FUNCTIONS.items():
        for c_index, (key, _, overrides) in enumerate(CONFIGS):
            dims = DIMENSIONS if key == "base" else (10,)
            for n_vars in dims:
                for run in range(RUNS):
                    seed = 10_000 * c_index + 100 * n_vars + run
                    cfg = replace(BASE, function=function, lower=lower, upper=upper, n_vars=n_vars, seed=seed, **overrides)
                    tasks.append((key, cfg, run))
    return tasks


def _execute(task):
    key, cfg, run = task
    result = build_algorithm(cfg).run()
    history = np.array([(s.best, s.mean, s.std, s.best_so_far) for s in result.history])
    return {
        "function": cfg.function, "config": key, "n_vars": cfg.n_vars, "run": run, "seed": cfg.seed,
        "best": result.best_value, "time": result.elapsed, "best_x": result.best_x, "history": history,
    }


def run_all() -> list[dict]:
    tasks = _tasks()
    print(f"Uruchomień: {len(tasks)}")
    results = []
    with ProcessPoolExecutor() as pool:
        for i, res in enumerate(pool.map(_execute, tasks, chunksize=4), 1):
            results.append(res)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(results))
    data = REPORT / "data"
    data.mkdir(parents=True, exist_ok=True)
    with open(data / "experiment_runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["function", "config", "n_vars", "run", "seed", "best_value", "time_s"])
        for r in results:
            writer.writerow([r["function"], r["config"], r["n_vars"], r["run"], r["seed"], r["best"], f"{r['time']:.4f}"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", action="store_true", help="tylko wykresy i tabele z zapisanych wyników")
    args = parser.parse_args()
    results = pickle.loads(CACHE.read_bytes()) if args.plots else run_all()
    from .report_assets import build_assets

    build_assets(results)


if __name__ == "__main__":
    main()
