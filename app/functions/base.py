from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np

# opfunu importuje przestarzałe pkg_resources, co generuje ostrzeżenie przy starcie.
warnings.filterwarnings("ignore", category=UserWarning, module="opfunu")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


class ObjectiveFunction(ABC):
    """Wspólny interfejs funkcji celu niezależny od biblioteki, z której pochodzi."""

    def __init__(self, name: str, n_vars: int):
        self.name = name
        self.n_vars = n_vars

    @property
    @abstractmethod
    def suggested_bounds(self) -> tuple[float, float]: ...

    @property
    @abstractmethod
    def optimum_value(self) -> float | None: ...

    @property
    @abstractmethod
    def optimum_point(self) -> np.ndarray | None: ...

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float: ...

    def evaluate_population(self, xs: np.ndarray) -> np.ndarray:
        return np.fromiter((self.evaluate(x) for x in xs), dtype=float, count=len(xs))


class BenchmarkFunctionAdapter(ObjectiveFunction):
    """Adapter funkcji z biblioteki `benchmark_functions` (L. Baronti)."""

    def __init__(self, name: str, n_vars: int, function_class):
        super().__init__(name, n_vars)
        self._function = function_class(n_dimensions=n_vars)

    @property
    def suggested_bounds(self):
        lower, upper = self._function.suggested_bounds()
        return float(lower[0]), float(upper[0])

    @property
    def optimum_value(self):
        return float(self._function.minimum().score)

    @property
    def optimum_point(self):
        return np.asarray(self._function.minimum().position, dtype=float)

    def evaluate(self, x):
        return float(self._function(x))


class CecFunctionAdapter(ObjectiveFunction):
    """Adapter funkcji CEC 2014 z biblioteki `opfunu`."""

    def __init__(self, name: str, n_vars: int, function_class):
        super().__init__(name, n_vars)
        self._function = function_class(ndim=n_vars)

    @property
    def suggested_bounds(self):
        lower, upper = self._function.bounds[0]
        return float(lower), float(upper)

    @property
    def optimum_value(self):
        return float(self._function.f_global)

    @property
    def optimum_point(self):
        return np.asarray(self._function.x_global, dtype=float)

    def evaluate(self, x):
        return float(self._function.evaluate(np.asarray(x, dtype=float)))
