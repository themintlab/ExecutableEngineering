# ExecutableEngineer

Numerical methods course content for ENGPHYS 3NM4, built as a Jupyter Book 2 project with the in-repo `Mint_NM` package.

## Repository layout

- `index.md` — landing page
- `Chapters/` — chapter notebooks and markdown content
- `Mint_NM/` — local Python package used by selected notebooks
- `myst.yml` — Jupyter Book 2 configuration and table of contents
- `.github/workflows/` — package checks, book build, and Pages deploy

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-book.txt
jupyter-book build --all --html
```

## Mint_NM notebook behavior

- Local development installs `Mint_NM` from `./Mint_NM` in editable mode.
- Colab notebooks install the published `Mint-NM` package from PyPI.
- Colab launch is configured through the book-level top bar, not per-page badges.

## CI commands

```bash
python -m build ./Mint_NM
python -c "from Mint_NM import RootFinderOpen, RootFinderClosed, init_model"
jupyter-book build --all --html
```

## Deployment

Pushes to `main` build the book and deploy the generated site to GitHub Pages.
