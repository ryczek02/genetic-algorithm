"""Punkt wejścia aplikacji: python main.py"""
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from gui import GeneticAlgorithmApp  # noqa: E402


def main() -> None:
    GeneticAlgorithmApp().mainloop()


if __name__ == "__main__":
    main()
