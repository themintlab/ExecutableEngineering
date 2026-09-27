import json

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/polynomial_interpolation.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# 1. Fix Colab badge in Cell 0
source_0 = nb['cells'][0]['source']
for i, line in enumerate(source_0):
    if '&nbsp;[![Open In Colab' in line:
        source_0[i] = line.replace('&nbsp;[![Open In Colab', '[![Open In Colab')

# 2. Convert eqnarray to aligned in all markdown cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        src = cell['source']
        for i, line in enumerate(src):
            if '\\begin{eqnarray}' in line:
                src[i] = line.replace('\\begin{eqnarray}', '$$\n\\begin{aligned}')
            if '\\end{eqnarray}' in line:
                src[i] = line.replace('\\end{eqnarray}', '\\end{aligned}\n$$')
            # Also replace &=& with &= for aligned compatibility
            if '&=&' in line:
                src[i] = line.replace('&=&', '&=')

# 3. Insert import cell at index 1
import_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\n",
        "import plotly.graph_objects as go\n",
        "import sympy as sp\n",
        "from numpy.polynomial import legendre\n",
        "\n",
        "try:\n",
        "    import executable_engineering as exe\n",
        "except ImportError:\n",
        "    %pip install -q executable_engineering\n",
        "    import executable_engineering as exe\n"
    ]
}
nb['cells'].insert(1, import_cell)

# 4. Update the code cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src_text = "".join(cell.get('source', []))
        
        # Plot 1: Lagrange basis polynomials
        if "Calculate the Lagrange basis polynomials" in src_text:
            cell['source'] = [
                "# Data points\n",
                "x = [0, .5, 2]\n",
                "y = [1, 3, 2]\n",
                "\n",
                "# Calculate the Lagrange basis polynomials\n",
                "n = len(x)\n",
                "P = []\n",
                "for i in range(n):\n",
                "  numerator = 1\n",
                "  denominator = 1\n",
                "  for j in range(n):\n",
                "    if i != j:\n",
                "      numerator = np.polymul(numerator, np.poly1d([1, -x[j]]))\n",
                "      denominator = denominator * (x[i] - x[j])\n",
                "  P.append(np.poly1d(np.polydiv(numerator, denominator)[0]))\n",
                "\n",
                "# Plot the Lagrange basis polynomials\n",
                "x_plot = np.linspace(-1, 3, 100)\n",
                "\n",
                "fig = go.Figure()\n",
                "for i in range(n):\n",
                "    y_plot = P[i](x_plot)\n",
                "    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f'P_{i+1}(x)'))\n",
                "\n",
                "fig.add_trace(go.Scatter(x=x, y=[1] * len(x), mode='markers', marker=dict(color='black'), showlegend=False))\n",
                "fig.add_trace(go.Scatter(x=x, y=[0] * len(x), mode='markers', marker=dict(color='red'), showlegend=False))\n",
                "\n",
                "fig.update_layout(title='Lagrange Basis Polynomials', xaxis_title='x', yaxis_title='P_i(x)')\n",
                "fig.show()\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None

        # Plot 2: Lagrange polynomial interpolation
        elif "Construct the Lagrange polynomial" in src_text:
            cell['source'] = [
                "# Construct the Lagrange polynomial\n",
                "L = np.poly1d(0)\n",
                "for i in range(n):\n",
                "  L = L + y[i] * P[i]\n",
                "\n",
                "# Plot the Lagrange polynomial\n",
                "y_plot = L(x_plot)\n",
                "\n",
                "fig = go.Figure()\n",
                "fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='L(x)'))\n",
                "fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data points', marker=dict(color='red')))\n",
                "fig.update_layout(title='Lagrange Polynomial Interpolation', xaxis_title='x', yaxis_title='L(x)')\n",
                "fig.show()\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None

        # Code 3: sympy
        elif "import sympy as sp" in src_text:
            cell['source'] = [
                "x = [0, 1, 2, 3, 4]\n",
                "y = [1, 3, 2, 5, 7]\n",
                "\n",
                "n = len(x)\n",
                "x_sym = sp.Symbol('x')\n",
                "\n",
                "L = 0\n",
                "for i in range(n):\n",
                "    term = y[i]\n",
                "    for j in range(n):\n",
                "        if i != j:\n",
                "            term *= (x_sym - x[j]) / (x[i] - x[j])\n",
                "    L += term\n",
                "\n",
                "print(L)\n",
                "\n",
                "print('which is an ugly way of writing out:')\n",
                "print(L.simplify())\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None

        # Plot 4: numpy Legendre
        elif "legendre.legfit" in src_text:
            cell['source'] = [
                "# Define the function\n",
                "def f(x):\n",
                "  return np.exp(-(x/2)**2)\n",
                "\n",
                "# Create x values for plotting\n",
                "x_toy = np.linspace(-6, 6, 100)\n",
                "y_toy = f(x_toy)\n",
                "\n",
                "# Sample 11 random points in the range -5 to 6\n",
                "x_d = np.arange(-5, 6, 1)\n",
                "y_d = f(x_d)\n",
                "\n",
                "# Interpolate using numpy Legendre\n",
                "coefficients = legendre.legfit(x_d, y_d, len(x_d) - 1)\n",
                "legendre_polynomial = legendre.Legendre(coefficients)\n",
                "\n",
                "# Create x values for plotting the interpolated polynomial\n",
                "x_interp = np.linspace(-5.5, 5.5, 200)\n",
                "y_interp = legendre_polynomial(x_interp)\n",
                "\n",
                "# Plot the original curve, sampled points, and interpolated polynomial\n",
                "fig = go.Figure()\n",
                "fig.add_trace(go.Scatter(x=x_toy, y=y_toy, name='exp(-(x/2)^2)'))\n",
                "fig.add_trace(go.Scatter(x=x_d, y=y_d, mode='markers', name='Sampled points', marker=dict(color='red')))\n",
                "fig.add_trace(go.Scatter(x=x_interp, y=y_interp, name='Legendre Interpolation'))\n",
                "fig.update_layout(title='Function, Sampled Points, and Legendre Interpolation', xaxis_title='x', yaxis_title='y')\n",
                "fig.show()\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None


with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated polynomial_interpolation.ipynb")
