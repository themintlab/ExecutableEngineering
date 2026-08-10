# ExecutableEngineer

Numerical methods course content for ENGPHYS 3NM4, built as a Jupyter Book 2.0 project using MyST as the rendering engine, with the in-repo `Mint_NM` package.

## Repository layout

- `index.md` — landing page
- `Chapters/` — chapter notebooks and markdown content
- `Mint_NM/` — local Python package used by selected notebooks
- `myst.yml` — Jupyter Book 2.0 configuration (MyST format)
- `.github/workflows/` — package checks, book build, and Pages deploy

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-book.txt
jupyter-book build --site
```

Then view the built site:
```bash
jupyter-book start .
```

This will serve the site at `http://localhost:3000` (or run `python -m http.server 8000 --directory _build/site/public/` to use Python's built-in server).

## Mint_NM notebook behavior

- Local development installs `Mint_NM` from `./Mint_NM` in editable mode.
- Colab notebooks install the published `Mint-NM` package from PyPI.
- Colab launch is configured through the book-level top bar, not per-page badges.

## CI/CD

The repository uses GitHub Actions for continuous integration and deployment:

- **Mint_NM checks** (`mint-nm-checks.yml`): Builds the package and runs smoke tests on every push and PR
- **Jupyter Book build** (`book-build.yml`): Builds the book on every push and PR to validate the build succeeds
- **GitHub Pages deployment** (`pages-deploy.yml`): Deploys to GitHub Pages on main branch pushes

### Local build commands

```bash
# Build the Jupyter Book site
jupyter-book build --site

# Clean build artifacts
jupyter-book clean .

# Start local dev server (after building)
jupyter-book start .

# Build with watch mode (auto-rebuild on file changes)
jupyter-book build --site --watch
```

### Mint_NM package testing

```bash
# Build the package
python -m build ./Mint_NM

# Test imports
( cd /tmp && python -c "from Mint_NM import RootFinderOpen, RootFinderClosed, OptimizerOpen, OptimizerClosed, OptimizerGrad, init_model" )
```

## Deployment

Pushes to `main` branch automatically trigger:
1. Book build validation
2. Package checks for Mint_NM
3. Deployment to GitHub Pages at: **https://themintlab.github.io/ExecutableEngineer/**

### Manual deployment

You can manually trigger the deployment via GitHub Actions web UI with the "workflow_dispatch" event.

## Technology Stack

- **Jupyter Book 2.0** - Building the book
- **MyST** - Markdown rendering engine (replaces old Jupyter Book 1.0)
- **Jupyter Notebooks** - Course content
- **Python 3.11** - Book build environment


