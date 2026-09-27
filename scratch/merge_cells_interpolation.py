import json

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/interpolation.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# Combine cells 2, 3, 4 into a single markdown cell
combined_source_1 = []
combined_source_1.extend(nb['cells'][2]['source'])
combined_source_1.append("\n\n")
combined_source_1.extend(nb['cells'][3]['source'])
combined_source_1.append("\n\n")
combined_source_1.extend(nb['cells'][4]['source'])

nb['cells'][2]['source'] = combined_source_1

# Combine cells 5, 7 into a single markdown cell
combined_source_2 = []
combined_source_2.extend(nb['cells'][5]['source'])
combined_source_2.append("\n\n")
combined_source_2.extend(nb['cells'][7]['source'])

# The new structure
new_cells = [
    nb['cells'][0],
    nb['cells'][1],
    nb['cells'][2], # Combined intro slide
]

# Set the source for the example slide
example_md_cell = nb['cells'][5]
example_md_cell['source'] = combined_source_2
new_cells.append(example_md_cell)

# Append the code cell
new_cells.append(nb['cells'][6])

nb['cells'] = new_cells

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Re-re-chunked interpolation.ipynb")
