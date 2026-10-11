from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from functions.base import ObjectiveFunction

from .config import GAConfig
from .crossover import CrossoverMethod
from .elitism import EliteStrategy
from .encoding import BinaryEncoder
from .inversion import InversionOperator
from .mutation import MutationMethod
from .selection import SelectionMethod


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    best: float
    mean: float
    std: float
    best_so_far: float
    best_x: np.ndarray


@dataclass
class RunResult:
    config: GAConfig
    function_name: str
    best_value: float
    best_x: np.ndarray
    elapsed: float
    bits_per_var: int
    history: list[EpochStats] = field(default_factory=list)
    stopped: bool = False


class GeneticAlgorithm:
    """Klasyczny algorytm genetyczny: selekcja, krzyżowanie, mutacja, inwersja, elitaryzm."""

    def __init__(
        self,
        config: GAConfig,
        objective: ObjectiveFunction,
        encoder: BinaryEncoder,
        selection: SelectionMethod,
        crossover: CrossoverMethod,
        mutation: MutationMethod,
        inversion: InversionOperator,
        elitism: EliteStrategy,
    ):
        self.config = config
        self.objective = objective
        self.encoder = encoder
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.inversion = inversion
        self.elitism = elitism
        self.rng = np.random.default_rng(config.seed)

    def _scores(self, values: np.ndarray) -> np.ndarray:
        """Przystosowanie w konwencji "im więcej, tym lepiej"."""
        return values if self.config.maximize else -values

    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.config.maximize else a < b

    def _breed(self, population: np.ndarray, pool: np.ndarray, count: int) -> np.ndarray:
        cfg, rng = self.config, self.rng
        children = []
        while len(children) < count:
            a, b = population[rng.choice(pool, size=2)]
            if rng.random() < cfg.crossover_prob:
                a, b = self.crossover.cross(a, b, rng)
            for child in (a, b):
                if rng.random() < cfg.mutation_prob:
                    child = self.mutation.mutate(child, rng)
                children.append(self.inversion.apply(child, rng))
        return np.array(children[:count], dtype=np.uint8)

    def run(
        self,
        on_epoch: Callable[[EpochStats], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> RunResult:
        cfg = self.config
        start = time.perf_counter()
        population = self.encoder.random_population(cfg.population_size, self.rng)
        best_value = -np.inf if cfg.maximize else np.inf
        best_x = np.zeros(cfg.n_vars)
        history: list[EpochStats] = []
        stopped = False

        for epoch in range(1, cfg.epochs + 1):
            xs = self.encoder.decode(population)
            values = self.objective.evaluate_population(xs)
            scores = self._scores(values)

            leader = int(np.argmax(scores))
            if self._is_better(values[leader], best_value):
                best_value, best_x = float(values[leader]), xs[leader].copy()

            stats = EpochStats(
                epoch=epoch,
                best=float(values[leader]),
                mean=float(values.mean()),
                std=float(values.std()),
                best_so_far=best_value,
                best_x=best_x,
            )
            history.append(stats)
            if on_epoch:
                on_epoch(stats)
            if should_stop and should_stop():
                stopped = True
                break
            if epoch == cfg.epochs:
                break

            elite = population[self.elitism.select(scores)]
            pool = self.selection.select(scores, self.rng)
            children = self._breed(population, pool, cfg.population_size - len(elite))
            population = np.vstack((elite, children)) if len(elite) else children

        return RunResult(
            config=cfg,
            function_name=self.objective.name,
            best_value=best_value,
            best_x=best_x,
            elapsed=time.perf_counter() - start,
            bits_per_var=self.encoder.bits_per_var,
            history=history,
            stopped=stopped,
        )
