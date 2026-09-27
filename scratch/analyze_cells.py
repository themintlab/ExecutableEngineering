import json

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/polynomial_interpolation.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    c_type = cell['cell_type']
    source = cell.get('source', [])
    header = source[0].strip() if source else ""
    # Just print the first 40 chars of the cell to understand structure
    print(f"Cell {i} ({c_type}): {header[:40].replace(chr(10), ' ')}")
