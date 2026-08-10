# Mint_NM

`Mint_NM` is the interactive numerical methods support package used by this book.

## Local development

From the repository root:

```bash
python -m pip install -r requirements-book.txt
```

That installs the in-repo package in editable mode so notebook changes use the local source.

## Standalone package checks

```bash
python -m pip install -e ./Mint_NM
python -m build ./Mint_NM
```
