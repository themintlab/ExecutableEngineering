import json
import os

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/interpolation.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# 1. Fix Colab badge in Cell 0
source_0 = nb['cells'][0]['source']
for i, line in enumerate(source_0):
    if '&nbsp;[![Open In Colab' in line:
        source_0[i] = line.replace('&nbsp;[![Open In Colab', '[![Open In Colab')

# 2. Insert import cell at index 1
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
nb['cells'].insert(1, import_cell)

# 3. Modify the code cell
code_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        if any('def f(x):' in line for line in cell.get('source', [])):
            code_idx = i
            break

if code_idx != -1:
    new_source = [
        "# Define the function\n",
        "def f(x):\n",
        "  return np.exp(-(x/2)**2)\n",
        "\n",
        "# Create x values for plotting\n",
        "x_toy = np.linspace(-6, 6, 100)\n",
        "y_toy = f(x_toy)\n",
        "\n",
        "# Sample 11 times at 1-interval intervals\n",
        "x_d = np.arange(-5, 6, 1)\n",
        "y_d = f(x_d)\n",
        "\n",
        "# Plot the function and sampled points\n",
        "fig = go.Figure()\n",
        "fig.add_trace(go.Scatter(x=x_toy, y=y_toy, name='exp(-(x/2)^2)'))\n",
        "fig.add_trace(go.Scatter(x=x_d, y=y_d, mode='markers', name='Sampled points', marker=dict(color='red')))\n",
        "fig.update_layout(title='Function and Sampled Points', xaxis_title='x', yaxis_title='y')\n",
        "fig.show()\n",
        "\n",
        "print(\"x_d:\", x_d)\n",
        "print(\"y_d:\", y_d)\n"
    ]
    nb['cells'][code_idx]['source'] = new_source
    nb['cells'][code_idx]['outputs'] = [] # Clear outputs since matplotlib is removed
    nb['cells'][code_idx]['execution_count'] = None

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated interpolation.ipynb")
