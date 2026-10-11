from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CrossoverMethod(ABC):
    @abstractmethod
    def cross(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Zwraca dwóch potomków powstałych z rodzicow `a` i `b`."""


def _swap_by_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.where(mask, b, a), np.where(mask, a, b)


class OnePointCrossover(CrossoverMethod):
    def cross(self, a, b, rng):
        point = rng.integers(1, len(a))
        return np.concatenate((a[:point], b[point:])), np.concatenate((b[:point], a[point:]))


class TwoPointCrossover(CrossoverMethod):
    def cross(self, a, b, rng):
        if len(a) < 3:
            return OnePointCrossover().cross(a, b, rng)
        p1, p2 = np.sort(rng.choice(np.arange(1, len(a)), size=2, replace=False))
        mask = np.zeros(len(a), dtype=bool)
        mask[p1:p2] = True
        return _swap_by_mask(a, b, mask)


class UniformCrossover(CrossoverMethod):
    """Każdy gen jest niezależnie wymieniany między rodzicami z prawdopodobieństwem 0.5."""

    def cross(self, a, b, rng):
        return _swap_by_mask(a, b, rng.random(len(a)) < 0.5)


class GranularCrossover(CrossoverMethod):
    """Krzyżowanie ziarniste: chromosom dzielony jest na ziarna (bloki) o długości
    `grain_size`, a każde ziarno potomek dziedziczy w całości od losowego rodzica."""

    def __init__(self, grain_size: int):
        self.grain_size = grain_size

    def cross(self, a, b, rng):
        n_grains = -(-len(a) // self.grain_size)
        grain_mask = rng.random(n_grains) < 0.5
        mask = np.repeat(grain_mask, self.grain_size)[: len(a)]
        return _swap_by_mask(a, b, mask)
