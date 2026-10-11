from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from functions import FUNCTIONS, create_function
from ga import GAConfig
from ga.encoding import BinaryEncoder
from ga.factory import CROSSOVERS, MUTATIONS, SELECTIONS

ELITE_MODES = {"percent": "Ułamek (0-1)", "count": "Liczba osobników"}


def _label_to_key(options: dict, label: str) -> str:
    return next(key for key, value in options.items() if (value[0] if isinstance(value, tuple) else value) == label)


class ParameterForm(ttk.Frame):
    """Formularz z pełną konfiguracją algorytmu. Zwraca gotowy obiekt GAConfig."""

    def __init__(self, master, defaults: GAConfig | None = None):
        super().__init__(master)
        self.columnconfigure((0, 1), weight=1)
        self._vars: dict[str, tk.StringVar] = {}
        self._widgets: dict[str, ttk.Widget] = {}
        self._build()
        self.set_config(defaults or GAConfig())

    # ------------------------------------------------------------------ budowa
    def _section(self, title: str, column: int) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(self, text=title, padding=(10, 2, 10, 6))
        row = len(self.grid_slaves(column=column))
        frame.grid(row=row, column=column, sticky="new", pady=(0, 8), padx=(0 if column == 0 else 12, 0))
        frame.columnconfigure(1, weight=1)
        return frame

    def _row(self, frame, key: str, label: str, widget_factory=None) -> ttk.Widget:
        row = frame.grid_size()[1]
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        var = self._vars.setdefault(key, tk.StringVar())
        widget = widget_factory(frame, var) if widget_factory else ttk.Entry(frame, textvariable=var, width=9)
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        self._widgets[key] = widget
        return widget

    @staticmethod
    def _combo(values):
        return lambda parent, var: ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=15)

    def _build(self) -> None:
        f = self._section("Funkcja celu", 0)
        names = [spec.display_name for spec in FUNCTIONS.values()]
        self._row(f, "function", "Funkcja", self._combo(names)).bind("<<ComboboxSelected>>", self._on_function_change)
        self._row(f, "n_vars", "Liczba zmiennych")
        self._row(f, "lower", "Początek zakresu")
        self._row(f, "upper", "Koniec zakresu")
        self._goal = tk.StringVar(value="min")
        goal = ttk.Frame(f)
        goal.grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Radiobutton(goal, text="Minimalizacja", variable=self._goal, value="min").pack(side="left", padx=(0, 16))
        ttk.Radiobutton(goal, text="Maksymalizacja", variable=self._goal, value="max").pack(side="left")
        self.optimum_hint = ttk.Label(f, style="Hint.TLabel", wraplength=330, justify="left")
        self.optimum_hint.grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w", pady=(4, 0))

        f = self._section("Reprezentacja i populacja", 0)
        self._row(f, "precision", "Dokładność")
        self._row(f, "population_size", "Wielkość populacji")
        self._row(f, "epochs", "Liczba epok")
        self.bits_hint = ttk.Label(f, style="Hint.TLabel")
        self.bits_hint.grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w", pady=(2, 0))

        f = self._section("Selekcja", 0)
        self._row(f, "selection", "Metoda selekcji", self._combo([v[0] for v in SELECTIONS.values()]))
        self._row(f, "selection_percentage", "Ułamek do selekcji")
        self._row(f, "tournament_size", "Rozmiar turnieju")

        f = self._section("Krzyżowanie, mutacja, inwersja", 1)
        self._row(f, "crossover", "Metoda krzyżowania", self._combo([v[0] for v in CROSSOVERS.values()]))
        self._row(f, "crossover_prob", "P. krzyżowania")
        self._row(f, "grain_size", "Rozmiar ziarna (bity)")
        self._row(f, "mutation", "Metoda mutacji", self._combo([v[0] for v in MUTATIONS.values()]))
        self._row(f, "mutation_prob", "P. mutacji")
        self._row(f, "inversion_prob", "P. inwersji")

        f = self._section("Strategia elitarna i losowość", 1)
        self._row(f, "elite_mode", "Tryb elity", self._combo(list(ELITE_MODES.values())))
        self._row(f, "elite_value", "Wielkość elity")
        self._row(f, "seed", "Ziarno RNG (opcjonalne)")

        for key in ("selection", "crossover"):
            self._vars[key].trace_add("write", lambda *_: self._refresh_dependent_fields())
        for key in ("lower", "upper", "precision", "n_vars", "function"):
            self._vars[key].trace_add("write", lambda *_: self._refresh_hints())

    # ------------------------------------------------------------- zachowanie
    def _function_key(self) -> str:
        label = self._vars["function"].get()
        return next(k for k, spec in FUNCTIONS.items() if spec.display_name == label)

    def _on_function_change(self, _event=None) -> None:
        spec = FUNCTIONS[self._function_key()]
        lower, upper = spec.default_bounds
        self._vars["lower"].set(f"{lower:g}")
        self._vars["upper"].set(f"{upper:g}")
        try:
            n_vars = int(self._vars["n_vars"].get())
        except ValueError:
            n_vars = 10
        if not spec.supports(n_vars):
            self._vars["n_vars"].set(str(spec.supported_dims[0]))

    def _refresh_dependent_fields(self) -> None:
        selection = self._vars["selection"].get()
        uses_percentage = selection in (SELECTIONS["best"][0], SELECTIONS["worst"][0])
        self._set_enabled("selection_percentage", uses_percentage)
        self._set_enabled("tournament_size", selection == SELECTIONS["tournament"][0])
        self._set_enabled("grain_size", self._vars["crossover"].get() == CROSSOVERS["granular"][0])

    def _set_enabled(self, key: str, enabled: bool) -> None:
        self._widgets[key].state(["!disabled"] if enabled else ["disabled"])

    def _refresh_hints(self) -> None:
        try:
            key = self._function_key()
            spec = FUNCTIONS[key]
            n_vars = int(self._vars["n_vars"].get())
            dims = "dowolna" if spec.supported_dims is None else ", ".join(map(str, spec.supported_dims))
            if spec.supports(n_vars):
                function = create_function(key, n_vars)
                point = function.optimum_point
                shown = ", ".join(f"{v:.4g}" for v in point[:3]) + (", ..." if len(point) > 3 else "")
                self.optimum_hint.config(
                    text=f"Minimum globalne: f* = {function.optimum_value:.6g} w x* = ({shown})\n"
                    f"Obsługiwana liczba zmiennych: {dims}"
                )
            else:
                self.optimum_hint.config(text=f"Obsługiwana liczba zmiennych: {dims}")
        except (ValueError, StopIteration):
            self.optimum_hint.config(text="")
        try:
            encoder = BinaryEncoder(
                float(self._num("lower")), float(self._num("upper")), float(self._num("precision")), int(self._num("n_vars"))
            )
            self.bits_hint.config(
                text=f"{encoder.bits_per_var} bitów na zmienną, chromosom: {encoder.length} bitów"
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            self.bits_hint.config(text="")

    def _num(self, key: str) -> str:
        return self._vars[key].get().strip().replace(",", ".")

    # ------------------------------------------------------------ konfiguracja
    def set_config(self, cfg: GAConfig) -> None:
        values = cfg.to_dict()
        for key, var in self._vars.items():
            if key == "function":
                var.set(FUNCTIONS[cfg.function].display_name)
            elif key == "selection":
                var.set(SELECTIONS[cfg.selection][0])
            elif key == "crossover":
                var.set(CROSSOVERS[cfg.crossover][0])
            elif key == "mutation":
                var.set(MUTATIONS[cfg.mutation][0])
            elif key == "elite_mode":
                var.set(ELITE_MODES[cfg.elite_mode])
            elif key == "seed":
                var.set("" if cfg.seed is None else str(cfg.seed))
            else:
                var.set(f"{values[key]:g}")
        self._goal.set("max" if cfg.maximize else "min")
        self._refresh_dependent_fields()
        self._refresh_hints()

    def get_config(self) -> GAConfig:
        """Parsuje formularz. Rzuca ValueError z czytelnym komunikatem."""
        labels = {
            "n_vars": "Liczba zmiennych", "lower": "Początek zakresu", "upper": "Koniec zakresu",
            "precision": "Dokładność", "population_size": "Wielkość populacji", "epochs": "Liczba epok",
            "selection_percentage": "Ułamek wybieranych", "tournament_size": "Rozmiar turnieju",
            "crossover_prob": "P. krzyżowania", "grain_size": "Rozmiar ziarna", "mutation_prob": "P. mutacji",
            "inversion_prob": "P. inwersji", "elite_value": "Wielkość elity", "seed": "Ziarno RNG",
        }
        ints = {"n_vars", "population_size", "epochs", "tournament_size", "grain_size"}
        parsed = {}
        for key, label in labels.items():
            raw = self._num(key)
            if key == "seed" and raw == "":
                parsed[key] = None
                continue
            try:
                parsed[key] = int(raw) if key in ints or key == "seed" else float(raw)
            except ValueError:
                kind = "liczbą całkowitą" if key in ints or key == "seed" else "liczbą"
                raise ValueError(f"Pole \"{label}\" musi być {kind}.") from None
        cfg = GAConfig(
            function=self._function_key(),
            maximize=self._goal.get() == "max",
            selection=_label_to_key(SELECTIONS, self._vars["selection"].get()),
            crossover=_label_to_key(CROSSOVERS, self._vars["crossover"].get()),
            mutation=_label_to_key(MUTATIONS, self._vars["mutation"].get()),
            elite_mode=_label_to_key(ELITE_MODES, self._vars["elite_mode"].get()),
            **parsed,
        )
        cfg.validate()
        create_function(cfg.function, cfg.n_vars)  # sprawdza obsługiwaną liczbę zmiennych
        return cfg

    def action_area(self) -> ttk.Frame:
        """Miejsce na przyciski pod ostatnią sekcją prawej kolumny formularza."""
        frame = ttk.Frame(self)
        frame.grid(row=len(self.grid_slaves(column=1)), column=1, sticky="ew", padx=(12, 0), pady=(4, 0))
        return frame

    def set_running(self, running: bool) -> None:
        for key, widget in self._widgets.items():
            if running:
                widget.state(["disabled"])
            else:
                widget.state(["!disabled"])
                if isinstance(widget, ttk.Combobox):
                    widget.state(["readonly"])
        if not running:
            self._refresh_dependent_fields()
