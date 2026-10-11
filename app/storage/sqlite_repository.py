from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ga.algorithm import RunResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    function    TEXT    NOT NULL,
    n_vars      INTEGER NOT NULL,
    best_value  REAL    NOT NULL,
    elapsed_s   REAL    NOT NULL,
    epochs_done INTEGER NOT NULL,
    stopped     INTEGER NOT NULL,
    best_x      TEXT    NOT NULL,
    config      TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS epochs (
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    epoch       INTEGER NOT NULL,
    best        REAL    NOT NULL,
    mean        REAL    NOT NULL,
    std         REAL    NOT NULL,
    best_so_far REAL    NOT NULL,
    best_x      TEXT    NOT NULL,
    PRIMARY KEY (run_id, epoch)
);
"""


class SqliteResultRepository:
    """Zapisuje wyniki do bazy SQLite.

    Tabela `runs` zawiera jeden wiersz podsumowania na uruchomienie, a tabela `epochs`
    wartości z każdej epoki danego uruchomienia.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_run(self, result: RunResult) -> int:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO runs (created_at, function, n_vars, best_value, elapsed_s, epochs_done, stopped, best_x, config)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    result.function_name,
                    result.config.n_vars,
                    float(result.best_value),
                    result.elapsed,
                    len(result.history),
                    int(result.stopped),
                    json.dumps(result.best_x.tolist()),
                    json.dumps(result.config.to_dict()),
                ),
            )
            run_id = cursor.lastrowid
            conn.executemany(
                "INSERT INTO epochs (run_id, epoch, best, mean, std, best_so_far, best_x) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (run_id, s.epoch, float(s.best), float(s.mean), float(s.std), float(s.best_so_far),
                     json.dumps(s.best_x.tolist()))
                    for s in result.history
                ),
            )
        return run_id
