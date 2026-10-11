from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SelectionMethod(ABC):
    """Wybiera pulę rodzicielska. `scores` jest zawsze "im więcej, tym lepiej"."""

    @abstractmethod
    def select(self, scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Zwraca indeksy osobników wybranych do puli rodzicielskiej."""


def _pool_size(percentage: float, population: int) -> int:
    return min(population, max(2, round(percentage * population)))


class BestSelection(SelectionMethod):
    """Do puli trafia zadany procent najlepszych osobników."""

    def __init__(self, percentage: float):
        self.percentage = percentage

    def select(self, scores, rng):
        return np.argsort(-scores)[: _pool_size(self.percentage, len(scores))]


class WorstSelection(SelectionMethod):
    """Do puli trafia zadany procent najsłabszych osobników (zwiększa różnorodność)."""

    def __init__(self, percentage: float):
        self.percentage = percentage

    def select(self, scores, rng):
        return np.argsort(scores)[: _pool_size(self.percentage, len(scores))]


class RouletteSelection(SelectionMethod):
    """Selekcja kołem ruletki: szansa wyboru proporcjonalna do przystosowania.

    Przystosowanie jest przesuwane tak, by najgorszy osobnik miał wagę bliską zeru,
    dzięki czemu metoda działa zarówno dla minimalizacji, jak i maksymalizacji.
    """

    def select(self, scores, rng):
        weights = scores - scores.min()
        total = weights.sum()
        n = len(scores)
        if not np.isfinite(total) or total <= 0:
            return rng.integers(0, n, size=n)
        return rng.choice(n, size=n, p=weights / total)


class TournamentSelection(SelectionMethod):
    """Selekcja turniejowa: zwycięzca losowej grupy `size` osobników trafia do puli."""

    def __init__(self, size: int):
        self.size = size

    def select(self, scores, rng):
        n = len(scores)
        contestants = rng.integers(0, n, size=(n, self.size))
        winners = np.argmax(scores[contestants], axis=1)
        return contestants[np.arange(n), winners]
