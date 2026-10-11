"""Eksperymenty do sprawozdania: każda konfiguracja uruchamiana 10 razy.

Uruchomienie (z katalogu app/):
    python -m experiments.run_experiments            # obliczenia + wykresy + tabele
    python -m experiments.run_experiments --plots    # tylko wykresy i tabele z zapisanych wyników

Uruchomienia wykonywane są kolejno w jednym procesie, żeby zmierzone czasy były porównywalne.
Wynik każdego uruchomienia trafia do bazy report/data/experiments.db.
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import warnings
from contextlib import closing
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
    for i, task in enumerate(tasks, 1):
        results.append(_execute(task))
        if i % 20 == 0:
            print(f"  {i}/{len(tasks)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(results))
    data = REPORT / "data"
    data.mkdir(parents=True, exist_ok=True)
    save_to_database(results, data / "experiments.db")
    return results


def save_to_database(results: list[dict], path: Path) -> None:
    path.unlink(missing_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TABLE runs (function TEXT, config TEXT, n_vars INTEGER, run INTEGER, seed INTEGER,"
            " best_value REAL, time_s REAL, PRIMARY KEY (function, config, n_vars, run))"
        )
        conn.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            ((r["function"], r["config"], r["n_vars"], r["run"], r["seed"], float(r["best"]), r["time"]) for r in results),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", action="store_true", help="tylko wykresy i tabele z zapisanych wyników")
    args = parser.parse_args()
    results = pickle.loads(CACHE.read_bytes()) if args.plots else run_all()
    from .report_assets import build_assets

    build_assets(results)


if __name__ == "__main__":
    main()
