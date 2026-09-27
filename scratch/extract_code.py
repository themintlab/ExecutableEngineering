import json

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/polynomial_interpolation.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

with open('/home/wellandm/Code/ExecutableEngineering/scratch/code_cells.py', 'w') as out:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            out.write(f"# --- CELL {i} ---\n")
            out.write("".join(cell.get('source', [])))
            out.write("\n\n")

print("Extracted code cells.")
