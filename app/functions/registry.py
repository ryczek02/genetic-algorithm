from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import benchmark_functions as bf
from opfunu.cec_based import cec2014

from .base import BenchmarkFunctionAdapter, CecFunctionAdapter, ObjectiveFunction

CEC_DIMENSIONS = (10, 20, 30, 50, 100)


@dataclass(frozen=True)
class FunctionSpec:
    key: str
    display_name: str
    group: str
    factory: Callable[[int], ObjectiveFunction]
    default_bounds: tuple[float, float]
    supported_dims: tuple[int, ...] | None = None  # None = dowolna liczba zmiennych

    def supports(self, n_vars: int) -> bool:
        return n_vars >= 1 and (self.supported_dims is None or n_vars in self.supported_dims)


def _benchmark(key, name, cls, bounds) -> FunctionSpec:
    return FunctionSpec(key, name, "Proste", lambda n: BenchmarkFunctionAdapter(name, n, cls), bounds)


def _cec(key, name, cls) -> FunctionSpec:
    return FunctionSpec(
        key, name, "CEC 2014", lambda n: CecFunctionAdapter(name, n, cls), (-100.0, 100.0), CEC_DIMENSIONS
    )


_SPECS = [
    _benchmark("ackley", "Ackley", bf.Ackley, (-32.768, 32.768)),
    _benchmark("schwefel", "Schwefel", bf.Schwefel, (-500.0, 500.0)),
    _cec("cec2014_f4", "CEC2014 F4 (Rosenbrock)", cec2014.F42014),
    _cec("cec2014_f8", "CEC2014 F8 (Rastrigin)", cec2014.F82014),
]

FUNCTIONS: dict[str, FunctionSpec] = {spec.key: spec for spec in _SPECS}


def create_function(key: str, n_vars: int) -> ObjectiveFunction:
    spec = FUNCTIONS[key]
    if not spec.supports(n_vars):
        dims = ", ".join(map(str, spec.supported_dims or ()))
        raise ValueError(f"Funkcja {spec.display_name} obsługuje tylko liczbę zmiennych: {dims}.")
    return spec.factory(n_vars)
