# Genetic Algorithm 

A classic genetic algorithm with binary chromosomes and a Tkinter GUI.

![Demo (3x speed)](docs/demo.gif)

Full video: [docs/prezentacja.mp4](docs/prezentacja.mp4)

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
python app/main.py
```

Python needs Tkinter. On macOS with Homebrew: `brew install python-tk@3.12`.

## Folders

- `app/` source code (GUI, algorithm, test functions)
- `app/results/` CSV results of every run
- `report/` LaTeX report and figures
