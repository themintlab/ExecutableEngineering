import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors

def visual_solve_2d(A, b, x_range=None, title="Linear System", labels=None):
    """
    Solves and visualizes a 2D linear system Ax = b.
    Plots the lines represented by the rows of A and b, and the solution point.
    Falls back to the pseudoinverse for singular or non-square systems.
    
    Args:
        A: (N, 2) array-like of coefficients.
        b: (N,) array-like of constants.
        x_range: Tuple of (xmin, xmax) for plotting. If None, auto-calculated.
        title: Plot title.
        labels: Tuple of (x_label, y_label) for axes.
    """
    if labels is None:
        labels = ("x", "y")
        
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    
    if A.shape[1] != 2:
        raise ValueError("visual_solve_2d only supports 2D systems (A must have 2 columns).")
    
    # 1. Solve the system using the pseudoinverse
    # For a square, full-rank matrix, pinv is mathematically identical to the standard inverse.
    # For singular or non-square matrices, it finds the least-squares/minimum-norm solution.
    sol = np.linalg.pinv(A) @ b
    
    # Check if the solution is exact AND unique.
    # It is exact if A*x = b. It is unique if the rank of A equals the number of variables (columns).
    is_exact = np.allclose(A @ sol, b)
    is_unique = np.linalg.matrix_rank(A) == A.shape[1]
    is_exact_unique = is_exact and is_unique
        
    # 2. Determine plotting range
    if x_range is None:
        center_x = sol[0]
        x_range = (center_x - 10, center_x + 10)
        
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    
    fig = go.Figure()
    
    colors = pcolors.qualitative.Plotly
    
    # 3. Plot each line
    for i in range(A.shape[0]):
        a1, a2 = A[i]
        bi = b[i]
        
        if np.abs(a2) > 1e-10:
            # y = (bi - a1*x) / a2
            y_vals = (bi - a1 * x_vals) / a2
            line_name = f"{a1:g}{labels[0]} + {a2:g}{labels[1]} = {bi:g}"
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=line_name, line=dict(color=colors[i % len(colors)])))
        else:
            # Vertical line: x = bi / a1
            if np.abs(a1) > 1e-10:
                x_val = bi / a1
                line_name = f"{a1:g}{labels[0]} = {bi:g}"
                fig.add_trace(go.Scatter(x=[x_val, x_val], y=[-100, 100], mode='lines', name=line_name, line=dict(color=colors[i % len(colors)])))
                
    # 4. Plot solution point
    if is_unique:
        if is_exact:
            sol_label = "Exact Solution"
        else:
            sol_label = "Best Fit Solution"
            
        hover_text = f"{labels[0]}={sol[0]:.4g}, {labels[1]}={sol[1]:.4g}"
        fig.add_trace(go.Scatter(
            x=[sol[0]], 
            y=[sol[1]], 
            mode='markers', 
            marker=dict(color='black', size=12, symbol='star'),
            name=sol_label,
            hoverinfo='text',
            hovertext=hover_text
        ))
    
    # Auto-adjust y-range nicely around the solution if not a vertical-only mess
    y_margin = (x_range[1] - x_range[0]) / 2.0
    
    fig.update_layout(
        title=title,
        xaxis_title=labels[0],
        yaxis_title=labels[1],
        yaxis=dict(range=[sol[1] - y_margin, sol[1] + y_margin]),
        template="plotly_white"
    )
    
    return fig
