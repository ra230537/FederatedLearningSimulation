# Repo Restructuring Design

**Date:** 2026-04-03  
**Status:** Approved

---

## Goal

Reorganize the repository into a clean `src/` layout, group analysis scripts and documentation, fix the `.gitignore` so experiment outputs are excluded from version control, and add a `pyproject.toml` to enable proper package resolution without path hacks.

---

## Final Folder Structure

```
FederatedLearningSimulation/
├── src/
│   ├── asynchronous/
│   │   ├── __init__.py        ← new (empty)
│   │   ├── client.py
│   │   ├── constants.py
│   │   ├── main.py
│   │   ├── monte_carlo.py
│   │   └── server.py
│   ├── synchronous/
│   │   ├── __init__.py        ← new (empty)
│   │   ├── client.py
│   │   ├── constants.py
│   │   ├── main.py
│   │   ├── monte_carlo.py
│   │   └── server.py
│   └── utils/
│       ├── __init__.py        ← new (empty)
│       ├── data_loader.py
│       ├── data_split.py
│       ├── models.py
│       └── plot_accuracy.py
├── experiments/
│   ├── ablation_study.py      ← subprocess paths updated
│   └── plot_ablation.py
├── docs/
│   ├── ablation_analysis.md
│   ├── results_analysis.md
│   ├── timeout_impact_analysis.md
│   └── receitas_rapidas.md
├── report/
│   ├── Relatório_Final_FAPESP.pdf   ← moved from root
│   ├── resultados.tex
│   ├── resultados.pdf
│   ├── relatorio.tex
│   └── relatorio.pdf
├── readme.md
├── requirements.txt
├── pyproject.toml             ← new
└── .gitignore                 ← rewritten
```

Files deleted: `email.md`  
Removed from git tracking: `synchronous/output/`, `asynchronous/output/`

---

## Import Strategy

### `pyproject.toml` (root)

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "federated-learning-sim"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["src"]
```

User runs `pip install -e .` once after cloning. This registers `utils`, `synchronous`, and `asynchronous` as top-level packages, making `from utils.data_loader import ...` work from any directory without `sys.path` manipulation.

### What changes in code

- **`src/synchronous/main.py`** and **`src/asynchronous/main.py`**: no import changes needed. Scripts are still run directly (`python src/synchronous/main.py`), so Python adds their directory to `sys.path` automatically — local imports (`from constants import *`, `from server import Server`, etc.) keep working. `from utils.xxx import ...` works via the installed package.

- **`experiments/ablation_study.py`**: two subprocess path strings updated:
  - `"asynchronous/main.py"` → `"src/asynchronous/main.py"`
  - `"plot_ablation.py"` → `"experiments/plot_ablation.py"`

- **`readme.md`**: CLI examples updated to reflect new paths:
  - `python synchronous/main.py` → `python src/synchronous/main.py`
  - `python asynchronous/main.py` → `python src/asynchronous/main.py`
  - `python ablation_study.py` → `python experiments/ablation_study.py`
  - `python plot_ablation.py` → `python experiments/plot_ablation.py`
  - `python -m utils.plot_accuracy` → `python -m utils.plot_accuracy` (unchanged — works after install)
  - Add `pip install -e .` to setup instructions

---

## `.gitignore` (full replacement)

```gitignore
# Python cache
__pycache__/
*.py[cod]

# Outputs de experimento (gerados em runtime)
output-*/
src/synchronous/output*/
src/asynchronous/output*/

# Dados de datasets
data/

# LaTeX — arquivos auxiliares gerados pelo compilador
*.aux
*.fdb_latexmk
*.fls
*.log
*.synctex.gz

# Ambientes locais
.venv/
venv/

# OS
.DS_Store
Thumbs.db

# IDEs
.vscode/
.idea/

# Contexto local do assistente
CLAUDE.md
```

---

## Git Cleanup

Remove currently-tracked output files:

```bash
git rm -r --cached synchronous/output/ asynchronous/output/
```

---

## Updated CLI (README)

```bash
# Setup (once)
pip install -e .

# Síncrono
python src/synchronous/main.py --dataset mnist --iid --percentile 50

# Assíncrono
python src/asynchronous/main.py --dataset cifar10 --non-iid --percentile 50 \
  --base-alpha 0.5 --decay-of-base-alpha 0.999 --tardiness-sensivity 0.1

# Ablação
python experiments/ablation_study.py --num-updates 40 --percentile 50 --num-clients 40

# Visualização
python -m utils.plot_accuracy --output-dir output-cifar-10 --non-iid
python experiments/plot_ablation.py --distribution all --vary all --percentile 50
```
