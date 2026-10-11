from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from ga.algorithm import RunResult

SUMMARY_FIELDS = [
    "run_id", "created_at", "function", "n_vars", "best_value", "elapsed_s", "epochs_done", "best_x", "config",
]
EPOCH_FIELDS = ["epoch", "best", "mean", "std", "best_so_far", "best_x"]


class CsvResultRepository:
    """Zapisuje wyniki do plików CSV.

    `runs.csv` zawiera jeden wiersz podsumowania na uruchomienie, a `run_<id>.csv`
    wartości z każdej epoki danego uruchomienia.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.directory / "runs.csv"

    def save_run(self, result: RunResult) -> str:
        created = datetime.now()
        run_id = f"{created:%Y%m%d_%H%M%S_%f}_{result.config.function}"

        with open(self.directory / f"run_{run_id}.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(EPOCH_FIELDS)
            for s in result.history:
                writer.writerow([s.epoch, s.best, s.mean, s.std, s.best_so_far, json.dumps(s.best_x.tolist())])

        is_new = not self.summary_path.exists()
        with open(self.summary_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(SUMMARY_FIELDS)
            writer.writerow([
                run_id,
                created.isoformat(timespec="seconds"),
                result.function_name,
                result.config.n_vars,
                result.best_value,
                f"{result.elapsed:.4f}",
                len(result.history),
                json.dumps(result.best_x.tolist()),
                json.dumps(result.config.to_dict()),
            ])
        return run_id
