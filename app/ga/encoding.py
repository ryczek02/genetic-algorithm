from __future__ import annotations

import math

import numpy as np


class BinaryEncoder:
    """Koduje wektor liczb rzeczywistych jako ciąg bitów o zadanej dokładności.

    Liczba bitów na zmienną to najmniejsze m spełniające (b - a) / (2^m - 1) <= dokładność.
    """

    def __init__(self, lower: float, upper: float, precision: float, n_vars: int):
        self.lower = float(lower)
        self.upper = float(upper)
        self.n_vars = int(n_vars)
        span = self.upper - self.lower
        self.bits_per_var = max(1, math.ceil(math.log2(span / precision + 1)))
        self.length = self.bits_per_var * self.n_vars
        self._max_int = 2.0 ** self.bits_per_var - 1
        self._weights = 2.0 ** np.arange(self.bits_per_var - 1, -1, -1)
        self._scale = span / self._max_int

    @property
    def effective_precision(self) -> float:
        return self._scale

    def random_population(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=(size, self.length), dtype=np.uint8)

    def decode(self, population: np.ndarray) -> np.ndarray:
        """Zamienia populację (n, length) na macierz rzeczywistych argumentów (n, n_vars)."""
        population = np.atleast_2d(population)
        genes = population.reshape(population.shape[0], self.n_vars, self.bits_per_var)
        integers = genes @ self._weights
        return self.lower + integers * self._scale
