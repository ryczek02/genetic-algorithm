from __future__ import annotations

from functions import create_function

from .algorithm import GeneticAlgorithm
from .config import GAConfig
from .crossover import GranularCrossover, OnePointCrossover, TwoPointCrossover, UniformCrossover
from .elitism import EliteStrategy
from .encoding import BinaryEncoder
from .inversion import InversionOperator
from .mutation import BoundaryMutation, OnePointMutation, TwoPointMutation
from .selection import BestSelection, RouletteSelection, TournamentSelection, WorstSelection

# klucz -> (etykieta w GUI, fabryka przyjmująca konfigurację)
SELECTIONS = {
    "best": ("Selekcja najlepszych", lambda c: BestSelection(c.selection_percentage)),
    "worst": ("Selekcja najgorszych", lambda c: WorstSelection(c.selection_percentage)),
    "roulette": ("Koło ruletki", lambda c: RouletteSelection()),
    "tournament": ("Selekcja turniejowa", lambda c: TournamentSelection(c.tournament_size)),
}

CROSSOVERS = {
    "one_point": ("Jednopunktowe", lambda c: OnePointCrossover()),
    "two_point": ("Dwupunktowe", lambda c: TwoPointCrossover()),
    "uniform": ("Jednorodne", lambda c: UniformCrossover()),
    "granular": ("Ziarniste", lambda c: GranularCrossover(c.grain_size)),
}

MUTATIONS = {
    "boundary": ("Brzegowa", lambda c: BoundaryMutation()),
    "one_point": ("Jednopunktowa", lambda c: OnePointMutation()),
    "two_point": ("Dwupunktowa", lambda c: TwoPointMutation()),
}


def build_algorithm(config: GAConfig) -> GeneticAlgorithm:
    config.validate()
    return GeneticAlgorithm(
        config=config,
        objective=create_function(config.function, config.n_vars),
        encoder=BinaryEncoder(config.lower, config.upper, config.precision, config.n_vars),
        selection=SELECTIONS[config.selection][1](config),
        crossover=CROSSOVERS[config.crossover][1](config),
        mutation=MUTATIONS[config.mutation][1](config),
        inversion=InversionOperator(config.inversion_prob),
        elitism=EliteStrategy(config.elite_value, config.elite_mode),
    )
