from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MutationMethod(ABC):
    @abstractmethod
    def mutate(self, chromosome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Zwraca zmutowaną kopię chromosomu."""


class BoundaryMutation(MutationMethod):
    """Mutacja brzegowa: negacja pierwszego albo ostatniego bitu chromosomu."""

    def mutate(self, chromosome, rng):
        result = chromosome.copy()
        index = 0 if rng.random() < 0.5 else len(result) - 1
        result[index] ^= 1
        return result


class OnePointMutation(MutationMethod):
    def mutate(self, chromosome, rng):
        result = chromosome.copy()
        result[rng.integers(len(result))] ^= 1
        return result


class TwoPointMutation(MutationMethod):
    def mutate(self, chromosome, rng):
        result = chromosome.copy()
        count = min(2, len(result))
        result[rng.choice(len(result), size=count, replace=False)] ^= 1
        return result
