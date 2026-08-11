# executable_engineering

`executable_engineering` is the interactive numerical methods support package used by this book.

## Local development

From the repository root:

```bash
python -m pip install -r requirements-book.txt
```

That installs the in-repo package in editable mode so notebook changes use the local source.

## Standalone package checks

```bash
python -m pip install -e ./executable_engineering
( cd /tmp && python -c "from executable_engineering import RootFinderOpen, RootFinderClosed" )
python -m build ./executable_engineering
```
