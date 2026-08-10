# Contributing

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-check.txt
```

## Common commands

```bash
python -m build ./Mint_NM
python -c "from Mint_NM import RootFinderOpen, RootFinderClosed, OptimizerOpen"
jupyter-book build --all --html
```

## Notebook package rule

Keep `Mint_NM` in this monorepo. Local notebook work should use the editable in-repo package; Colab notebooks install the published package from PyPI.

## Git hygiene

Do not commit `_build/`, virtualenvs, notebook checkpoints, or other generated artifacts.
