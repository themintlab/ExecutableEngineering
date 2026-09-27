import json

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/radial_basis_functions.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# 1. Fix Colab badge in Cell 0
source_0 = nb['cells'][0]['source']
for i, line in enumerate(source_0):
    if '&nbsp;[![Open In Colab' in line:
        source_0[i] = line.replace('&nbsp;[![Open In Colab', '[![Open In Colab')

# 2. Convert eqnarray to aligned in all markdown cells (just in case)
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        src = cell['source']
        for i, line in enumerate(src):
            if '\\begin{eqnarray}' in line:
                src[i] = line.replace('\\begin{eqnarray}', '$$\n\\begin{aligned}')
            if '\\end{eqnarray}' in line:
                src[i] = line.replace('\\end{eqnarray}', '\\end{aligned}\n$$')
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
        
        # Cell 5: plot kernels
        if "Calculate the function values for each kernel" in src_text:
            cell['source'] = [
                "# Define the radial basis functions\n",
                "def gaussian(r, epsilon):\n",
                "  return np.exp(-(epsilon * r)**2)\n",
                "\n",
                "def inverse_quadratic(r, epsilon):\n",
                "  return 1 / (1 + (epsilon * r)**2)\n",
                "\n",
                "def inverse_multiquadric(r, epsilon):\n",
                "  return 1 / np.sqrt(1 + (epsilon * r)**2)\n",
                "\n",
                "epsilon = 1\n",
                "\n",
                "# Create a range of r values\n",
                "r_values = np.linspace(0, 10, 100)\n",
                "\n",
                "# Calculate the function values for each kernel\n",
                "gaussian_values = gaussian(r_values, epsilon)\n",
                "inverse_quadratic_values = inverse_quadratic(r_values, epsilon)\n",
                "inverse_multiquadric_values = inverse_multiquadric(r_values, epsilon)\n",
                "\n",
                "# Plot the radial basis functions\n",
                "fig = go.Figure()\n",
                "fig.add_trace(go.Scatter(x=r_values, y=gaussian_values, name='Gaussian'))\n",
                "fig.add_trace(go.Scatter(x=r_values, y=inverse_quadratic_values, name='Inverse Quadratic'))\n",
                "fig.add_trace(go.Scatter(x=r_values, y=inverse_multiquadric_values, name='Inverse Multiquadric'))\n",
                "fig.update_layout(title='Radial Basis Functions (epsilon = 1)', xaxis_title='r', yaxis_title='φ(r)')\n",
                "fig.show()\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None

        # Cell 9: toy function
        elif "y_d = f(x_d)" in src_text and "Sample 11 times" in src_text:
            cell['source'] = [
                "# Define the function\n",
                "def f(x):\n",
                "  return np.exp(-(x/2)**2)\n",
                "\n",
                "def gaussian(r, epsilon):\n",
                "  return np.exp(-(epsilon * r)**2)\n",
                "\n",
                "# Create x values for plotting\n",
                "x_toy = np.linspace(-6, 6, 100)\n",
                "y_toy = f(x_toy)\n",
                "\n",
                "# Sample 11 times at 1-interval intervals\n",
                "x_d = np.arange(-5, 6, 1)\n",
                "y_d = f(x_d)\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None

        # Cell 10: RBF interpolation
        elif "phi_matrix = np.zeros((len(x_d), len(x_d)))" in src_text:
            cell['source'] = [
                "# Create a matrix of the radial basis functions\n",
                "phi_matrix = np.zeros((len(x_d), len(x_d)))\n",
                "\n",
                "epsilon = 1\n",
                "\n",
                "for i in range(len(x_d)):\n",
                "  for j in range(len(x_d)):\n",
                "    phi_matrix[i, j] = gaussian(np.abs(x_d[i] - x_d[j]), epsilon)\n",
                "\n",
                "#~~ How do we solve for w_i?\n",
                "# Take a look at the matrix!\n",
                "\n",
                "np.set_printoptions(precision=2, suppress=True)\n",
                "print(phi_matrix)\n",
                "weights = np.linalg.solve(phi_matrix, y_d)\n",
                "\n",
                "# Define the interpolation function\n",
                "def interpolation_function(x, weights, x_d, epsilon):\n",
                "  y = 0\n",
                "  for i in range(len(x_d)):\n",
                "    y += weights[i] * gaussian(np.abs(x - x_d[i]), epsilon)\n",
                "  return y\n",
                "\n",
                "# Interpolate y_fit\n",
                "y_fit = [interpolation_function(x, weights, x_d, epsilon) for x in x_toy]\n",
                "\n",
                "# Plot the results\n",
                "fig = go.Figure()\n",
                "fig.add_trace(go.Scatter(x=x_toy, y=y_toy, name='Original Function'))\n",
                "fig.add_trace(go.Scatter(x=x_d, y=y_d, mode='markers', name='Data Points', marker=dict(color='red')))\n",
                "fig.add_trace(go.Scatter(x=x_toy, y=y_fit, name='Interpolation', line=dict(dash='dash')))\n",
                "fig.update_layout(title='Radial Basis Function Interpolation (Gaussian Kernel)', xaxis_title='x', yaxis_title='y')\n",
                "fig.show()\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None

        # Cell 13: 3D Plot
        elif "import plotly.graph_objects as go" in src_text:
            cell['source'] = [
                "# Define the target function\n",
                "def f(x, y):\n",
                "    return np.exp(-.5*x**2 - .2*y**2) * np.sin(x) * np.cos(.5*y)\n",
                "\n",
                "x_grid_vals = np.linspace(-3, 3, 100)\n",
                "y_grid_vals = np.linspace(-3, 3, 100)\n",
                "X, Y = np.meshgrid(x_grid_vals, y_grid_vals)\n",
                "Z_true = f(X, Y)\n",
                "\n",
                "num_samples = 100\n",
                "x_samples = np.random.uniform(-3, 3, num_samples)\n",
                "y_samples = np.random.uniform(-3, 3, num_samples)\n",
                "z_samples = f(x_samples, y_samples)\n",
                "\n",
                "def gaussian_rbf(r, epsilon):\n",
                "    return np.exp(-(epsilon * r)**2)\n",
                "\n",
                "dist_matrix = np.sqrt((x_samples[:, np.newaxis] - x_samples)**2 + (y_samples[:, np.newaxis] - y_samples)**2)\n",
                "average_distance = np.mean(dist_matrix)\n",
                "eps = 1 / average_distance\n",
                "print(f\"Using epsilon = {eps:.4f}\")\n",
                "\n",
                "phi_matrix = gaussian_rbf(dist_matrix, eps)\n",
                "\n",
                "print(f'The matrix condition number is, {np.linalg.cond(phi_matrix):.2e}')\n",
                "\n",
                "weights = np.linalg.solve(phi_matrix, z_samples)\n",
                "\n",
                "Z_fit = np.zeros_like(X)\n",
                "for i in range(num_samples):\n",
                "    r = np.sqrt((X - x_samples[i])**2 + (Y - y_samples[i])**2)\n",
                "    Z_fit += weights[i] * gaussian_rbf(r, eps)\n",
                "\n",
                "surface_true = go.Surface(x=X, y=Y, z=Z_true, colorscale='Blues', showscale=False, name='Original')\n",
                "surface_fit = go.Surface(x=X, y=Y, z=Z_fit, colorscale='Viridis', showscale=False, name='RBF Fit')\n",
                "scatter_samples = go.Scatter3d(\n",
                "    x=x_samples, y=y_samples, z=z_samples,\n",
                "    mode='markers', marker=dict(color='red', size=3), name='Samples'\n",
                ")\n",
                "\n",
                "fig_original = go.Figure(data=[surface_true, scatter_samples])\n",
                "fig_original.update_layout(\n",
                "    title='Original Function and Data Samples',\n",
                "    scene=dict(\n",
                "        xaxis_title='x', yaxis_title='y', zaxis_title='z',\n",
                "        zaxis=dict(range=[-.7, .7]),\n",
                "        aspectratio=dict(x=1, y=1, z=0.5)\n",
                "    )\n",
                ")\n",
                "print(\"Displaying the original function...\")\n",
                "fig_original.show()\n",
                "\n",
                "fig_fit = go.Figure(data=[surface_fit, scatter_samples])\n",
                "fig_fit.update_layout(\n",
                "    title='RBF Interpolated Fit and Data Samples',\n",
                "    scene=dict(\n",
                "        xaxis_title='x', yaxis_title='y', zaxis_title='z',\n",
                "        zaxis=dict(range=[-.7, .7]),\n",
                "        aspectmode='manual',\n",
                "        aspectratio=dict(x=1, y=1, z=0.5)\n",
                "    )\n",
                ")\n",
                "print(\"Displaying the RBF fitted function...\")\n",
                "fig_fit.show()\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated radial_basis_functions.ipynb")
