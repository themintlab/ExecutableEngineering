import numpy as np

def weierstrass(x, a=0.5, b=13, iterations=100):
    """
    Computes the Weierstrass function.
    
    Parameters:
    x (array-like): The points at which to evaluate the function.
    a (float): Parameter 0 < a < 1.
    b (float): Parameter such that ab > 1 + 3pi/2.
    iterations (int): Number of terms in the sum.
    
    Returns:
    y (ndarray): The function values.
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    for n in range(iterations):
        y += (a**n) * np.cos((b**n) * np.pi * x)
    return y

import plotly.graph_objects as go
from scipy.optimize import root

def plot_and_find_roots(func):
    x = np.linspace(-10, 10, 400)
    y = func(x)

    # Find roots
    x0s = np.arange(-10, 10, 1)
    roots = []
    for x0 in x0s:
        r = root(func, x0=x0)
        if r.success:
            roots.append((r.x[0], r.fun[0]))

    # Remove duplicate roots (within tolerance)
    unique_roots = []
    tol = 1e-6
    for rx, ry in roots:
        if not any(abs(rx - ux) < tol for ux, _ in unique_roots):
            unique_roots.append((rx, ry))

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)'))
    fig.add_hline(y=0, line_dash='dash', line_color='black', annotation_text='x-axis')
    if unique_roots:
        rx_vals = [rx for rx, _ in unique_roots]
        fig.add_trace(go.Scatter(x=rx_vals, y=[0]*len(unique_roots), mode='markers', marker=dict(color='red', size=10), name='Roots'))
    fig.update_layout(title='Plot of f(x) and its Roots', xaxis_title='x', yaxis_title='f(x)', xaxis_range=[-10, 10], yaxis_range=[-10, 10])
    fig.show()

def complex_plot_with_shadow(func, title='Complex Plot of f(z)', roots=None, plot_range=(-3, 3)):
    real_range = np.linspace(plot_range[0], plot_range[1], 100)
    imag_range = np.linspace(plot_range[0], plot_range[1], 100)
    abs_val = np.empty((len(real_range), len(imag_range)))

    for i, real in enumerate(real_range):
        for j, imag in enumerate(imag_range):
            z = complex(real, imag)
            result = func(z)
            abs_val[i, j] = abs(result)

    surface = go.Surface(
        z=abs_val.T, 
        x=real_range, 
        y=imag_range, 
        colorscale='Viridis',
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    )
    
    data = [surface]
    
    if roots is not None:
        root_x = [r.real for r in roots]
        root_y = [r.imag for r in roots]
        # Calculate z for roots (which is abs(func(r)) = 0)
        root_z = [0 for r in roots]
        
        data.append(go.Scatter3d(
            x=root_x, y=root_y, z=root_z, 
            mode='markers', 
            marker=dict(color='red', size=5, symbol='circle'),
            name='Roots'
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title, 
        scene=dict(xaxis_title='Real', yaxis_title='Imaginary', zaxis_title='|f(z)|')
    )
    fig.show()


from scipy.linalg import companion

def plot_polynomial_roots_companion(n):
    """
    Creates a random polynomial of order n, solves for its roots 
    using the companion matrix method, and plots the polynomial 
    along with the roots as a complex surface with a shadow.
    """
    # Create random coefficients from -10 to 10
    # A polynomial of order n has n+1 coefficients. Let's make the leading coefficient non-zero.
    coeffs = np.random.uniform(-10, 10, n + 1)
    
    # Make sure leading coefficient is not too small
    if abs(coeffs[0]) < 1:
        coeffs[0] = 1.0 * np.sign(coeffs[0]) if coeffs[0] != 0 else 1.0
        
    poly_func = np.poly1d(coeffs)
    
    print("Polynomial:")
    print(poly_func)
    
    # Solve for roots using companion matrix
    comp_matrix = companion(coeffs)
    
    print("\nCompanion Matrix:")
    print(np.round(comp_matrix, 2))
    
    roots = np.linalg.eigvals(comp_matrix)
    
    print("\nRoots from eigenvalues of companion matrix:")
    print(np.round(roots, 2))
    
    # Determine plot range based on roots
    max_abs_root = max(abs(roots)) if len(roots) > 0 else 1.0
    plot_limit = max_abs_root * 1.5
    
    # Plot using complex_plot_with_shadow
    title = f"Roots of a Random Order {n} Polynomial"
    complex_plot_with_shadow(poly_func, title=title, roots=roots, plot_range=(-plot_limit, plot_limit))

