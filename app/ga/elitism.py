from __future__ import annotations

import numpy as np


class EliteStrategy:
    """Przenosi najlepsze osobniki bez zmian do kolejnej populacji.

    `mode == "percent"`: `value` to ułamek populacji (np. 0.05 = 5%).
    `mode == "count"`: `value` to liczba osobników.
    """

    def __init__(self, value: float, mode: str):
        self.value = value
        self.mode = mode

    def size(self, population_size: int) -> int:
        if self.mode == "percent":
            count = round(self.value * population_size)
        else:
            count = int(self.value)
        return max(0, min(count, population_size - 2))

    def select(self, scores: np.ndarray) -> np.ndarray:
        return np.argsort(-scores)[: self.size(len(scores))]
