import json

def process_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    new_cells = []
    current_md_source = []
    current_md_metadata = {}
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = cell.get('source', [])
            if not source:
                continue
                
            header_line = source[0].strip()
            # It starts with a heavy header if it is `# ` or `## `
            starts_with_heavy = header_line.startswith('# ') or header_line.startswith('## ')
            
            # Special case for colab badge cell: it usually has the title
            if '[![Open In Colab' in "".join(source):
                starts_with_heavy = True
                
            if starts_with_heavy:
                if current_md_source:
                    new_cells.append({
                        'cell_type': 'markdown',
                        'metadata': current_md_metadata,
                        'source': current_md_source
                    })
                current_md_source = source
                current_md_metadata = cell.get('metadata', {})
            else:
                if current_md_source:
                    current_md_source.append("\n\n")
                    current_md_source.extend(source)
                else:
                    current_md_source = source
                    current_md_metadata = cell.get('metadata', {})
        else:
            # Code cell
            if current_md_source:
                new_cells.append({
                    'cell_type': 'markdown',
                    'metadata': current_md_metadata,
                    'source': current_md_source
                })
                current_md_source = []
            
            new_cells.append(cell)
            
    if current_md_source:
        new_cells.append({
            'cell_type': 'markdown',
            'metadata': current_md_metadata,
            'source': current_md_source
        })

    nb['cells'] = new_cells
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Processed {path}")

process_notebook('/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/polynomial_interpolation.ipynb')
process_notebook('/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/radial_basis_functions.ipynb')
