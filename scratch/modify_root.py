import json

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation_and_curve_fitting.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# Fix cell 0
source = nb['cells'][0]['source']
for i, line in enumerate(source):
    if '&nbsp;[![Open In Colab' in line:
        source[i] = line.replace('&nbsp;[![Open In Colab', '[![Open In Colab')

# Create import cell
import_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\n",
        "import plotly.graph_objects as go\n",
        "\n",
        "try:\n",
        "    import executable_engineering as exe\n",
        "except ImportError:\n",
        "    %pip install -q executable_engineering\n",
        "    import executable_engineering as exe\n"
    ]
}

# Insert at index 1
nb['cells'].insert(1, import_cell)

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Modified root notebook.")
