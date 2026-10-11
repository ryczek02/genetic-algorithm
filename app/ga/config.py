from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GAConfig:
    """Pełna konfiguracja pojedynczego uruchomienia algorytmu genetycznego."""

    function: str = "ackley"
    n_vars: int = 10
    lower: float = -32.768
    upper: float = 32.768
    maximize: bool = False

    precision: float = 1e-4
    population_size: int = 100
    epochs: int = 500

    selection: str = "tournament"
    selection_percentage: float = 0.3
    tournament_size: int = 3

    crossover: str = "two_point"
    crossover_prob: float = 0.8
    grain_size: int = 4

    mutation: str = "one_point"
    mutation_prob: float = 0.2
    inversion_prob: float = 0.05

    elite_value: float = 0.05
    elite_mode: str = "percent"  # "percent" albo "count"

    seed: int | None = None

    def validate(self) -> None:
        errors: list[str] = []
        if self.n_vars < 1:
            errors.append("Liczba zmiennych musi być >= 1.")
        if self.lower >= self.upper:
            errors.append("Początek zakresu musi być mniejszy niż koniec.")
        if self.precision <= 0:
            errors.append("Dokładność musi być dodatnia.")
        if self.population_size < 4:
            errors.append("Populacja musi liczyć co najmniej 4 osobniki.")
        if self.epochs < 1:
            errors.append("Liczba epok musi być >= 1.")
        for label, value in (
            ("krzyżowania", self.crossover_prob),
            ("mutacji", self.mutation_prob),
            ("inwersji", self.inversion_prob),
        ):
            if not 0.0 <= value <= 1.0:
                errors.append(f"Prawdopodobieństwo {label} musi należeć do [0, 1].")
        if not 0.0 < self.selection_percentage <= 1.0:
            errors.append("Procent selekcji musi należeć do (0, 1].")
        if self.tournament_size < 2 or self.tournament_size > self.population_size:
            errors.append("Rozmiar turnieju musi należeć do [2, populacja].")
        if self.grain_size < 1:
            errors.append("Rozmiar ziarna musi być >= 1.")
        if self.elite_mode not in ("percent", "count"):
            errors.append("Nieznany tryb strategii elitarnej.")
        elif self.elite_mode == "percent" and not 0.0 <= self.elite_value < 1.0:
            errors.append("Procent elity musi należeć do [0, 1).")
        elif self.elite_mode == "count" and not 0 <= self.elite_value < self.population_size:
            errors.append("Liczba osobników elitarnych musi być mniejsza od populacji.")
        if errors:
            raise ValueError("\n".join(errors))

    def to_dict(self) -> dict:
        return asdict(self)
