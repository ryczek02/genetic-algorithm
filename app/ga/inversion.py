from __future__ import annotations

import numpy as np


class InversionOperator:
    """Odwraca kolejność bitów pomiędzy dwoma losowymi punktami chromosomu."""

    def __init__(self, probability: float):
        self.probability = probability

    def apply(self, chromosome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if len(chromosome) < 2 or rng.random() >= self.probability:
            return chromosome
        i, j = np.sort(rng.choice(len(chromosome), size=2, replace=False))
        result = chromosome.copy()
        result[i : j + 1] = result[i : j + 1][::-1]
        return result
