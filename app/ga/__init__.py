"""Klasyczny algorytm genetyczny z binarną reprezentacją chromosomu."""
from .config import GAConfig
from .algorithm import GeneticAlgorithm, EpochStats, RunResult
from .factory import build_algorithm

__all__ = ["GAConfig", "GeneticAlgorithm", "EpochStats", "RunResult", "build_algorithm"]
