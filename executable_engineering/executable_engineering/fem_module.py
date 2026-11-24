import numpy as np
from scipy.optimize import root
import plotly.graph_objects as go


# =====================================================================
# 1. Pedagogical 1D Heat Equation FEM Solvers
#    Exposed in notebooks via {literalinclude} and exe.display_source()
# =====================================================================


def fem_heat_1d_linear(L=10.0, n_elements=20, dt=0.1, t_final=10.0):
    x_nodes = np.concatenate((
        np.linspace(0.0, 2.0, n_elements // 2 + 1),
        np.linspace(2.0 + (L - 2.0) / (n_elements // 2), L, n_elements // 2),
    ))
    n = len(x_nodes)
    K = np.zeros((n, n))
    M = np.zeros((n, n))

    for i in range(n - 1):
        h = x_nodes[i + 1] - x_nodes[i]
        K[i:i + 2, i:i + 2] += np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        M[i:i + 2, i:i + 2] += np.array([[2.0, 1.0], [1.0, 2.0]]) * h / 6.0

    K[-1, :] = 0.0
    M[-1, :] = 0.0
    M[-1, -1] = 1.0

    T = np.zeros(n)
    A = M + dt * K
    for _ in range(int(round(t_final / dt))):
        b = M @ T
        b[0] += dt * 1.0
        T = np.linalg.solve(A, b)

    return x_nodes, T, M, K


def fem_heat_1d_nonlinear_lagged(L=10.0, n_elements=20, dt=0.1, t_final=10.0):
    x_nodes = np.concatenate((
        np.linspace(0.0, 2.0, n_elements // 2 + 1),
        np.linspace(2.0 + (L - 2.0) / (n_elements // 2), L, n_elements // 2),
    ))
    n = len(x_nodes)

    def assemble(T):
        K = np.zeros((n, n))
        M = np.zeros((n, n))
        for i in range(n - 1):
            h = x_nodes[i + 1] - x_nodes[i]
            lam = 1.0 + T[i:i + 2].mean() / 10.0
            K[i:i + 2, i:i + 2] += np.array([[1.0, -1.0], [-1.0, 1.0]]) * lam / h
            M[i:i + 2, i:i + 2] += np.array([[2.0, 1.0], [1.0, 2.0]]) * h / 6.0
        K[-1, :] = 0.0
        M[-1, :] = 0.0
        M[-1, -1] = 1.0
        return K, M

    T = np.zeros(n)
    for _ in range(int(round(t_final / dt))):
        K, M = assemble(T)
        b = M @ T
        b[0] += dt * 1.0
        T = np.linalg.solve(M + dt * K, b)

    return x_nodes, T


def fem_heat_1d_nonlinear_root(L=10.0, n_elements=20, dt=0.1, t_final=10.0):
    x_nodes = np.concatenate((
        np.linspace(0.0, 2.0, n_elements // 2 + 1),
        np.linspace(2.0 + (L - 2.0) / (n_elements // 2), L, n_elements // 2),
    ))
    n = len(x_nodes)
    M = np.zeros((n, n))
    for i in range(n - 1):
        h = x_nodes[i + 1] - x_nodes[i]
        M[i:i + 2, i:i + 2] += np.array([[2.0, 1.0], [1.0, 2.0]]) * h / 6.0
    M[-1, :] = 0.0
    M[-1, -1] = 1.0

    def solve_step(T_prev):
        def residual(T):
            K = np.zeros((n, n))
            for i in range(n - 1):
                h = x_nodes[i + 1] - x_nodes[i]
                lam = 1.0 + 0.5 * (T[i] + T[i + 1]) / 10.0
                K[i:i + 2, i:i + 2] += np.array([[1.0, -1.0], [-1.0, 1.0]]) * lam / h
            K[-1, :] = 0.0
            b = M @ T_prev
            b[0] += dt * 1.0
            b[-1] = 0.0
            return (M @ T) + dt * (K @ T) - b

        return root(residual, T_prev).x

    T = np.zeros(n)
    for _ in range(int(round(t_final / dt))):
        T = solve_step(T)

    return x_nodes, T


def plot_matrix_sparsity(matrix, title="Sparsity Pattern of Mass Matrix M"):
    mat = np.asarray(matrix)
    rows, cols = np.nonzero(np.abs(mat) > 1e-14)
    vals = mat[rows, cols]
    n_rows, n_cols = mat.shape

    fig = go.Figure(
        data=go.Scatter(
            x=cols,
            y=rows,
            mode="markers",
            marker=dict(symbol="square", size=9, color="#2563EB"),
            customdata=vals,
            hovertemplate="Row %{y}, Col %{x}<br>Value: %{customdata:.4g}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=460,
        height=460,
        xaxis=dict(
            title="Column index j",
            range=[-0.8, n_cols - 0.2],
            constrain="domain",
            showgrid=True,
        ),
        yaxis=dict(
            title="Row index i",
            range=[n_rows - 0.2, -0.8],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
        ),
    )
    return fig


def _fem_heat_1d_snapshots(mode, L, n_elements, dt, t_values):
    x_nodes = np.concatenate((
        np.linspace(0.0, 2.0, n_elements // 2 + 1),
        np.linspace(2.0 + (L - 2.0) / (n_elements // 2), L, n_elements // 2),
    ))
    n = len(x_nodes)
    h = np.diff(x_nodes)

    M = np.zeros((n, n))
    for i in range(n - 1):
        M[i:i + 2, i:i + 2] += np.array([[2.0, 1.0], [1.0, 2.0]]) * h[i] / 6.0
    M[-1, :] = 0.0
    M[-1, -1] = 1.0

    def assemble_K(lam):
        K = np.zeros((n, n))
        c = lam / h
        for i in range(n - 1):
            K[i:i + 2, i:i + 2] += np.array([[1.0, -1.0], [-1.0, 1.0]]) * c[i]
        K[-1, :] = 0.0
        return K

    step_targets = sorted({int(round(t / dt)) for t in t_values})
    max_step = step_targets[-1]
    target_set = set(step_targets)

    T = np.zeros(n)
    snapshots = {}
    if 0 in target_set:
        snapshots[0] = T.copy()

    if mode == "linear":
        A = M + dt * assemble_K(np.ones(n - 1))
        for step in range(1, max_step + 1):
            b = M @ T
            b[0] += dt * 1.0
            T = np.linalg.solve(A, b)
            if step in target_set:
                snapshots[step] = T.copy()
    elif mode == "nonlinear_lagged":
        for step in range(1, max_step + 1):
            lam = 1.0 + 0.5 * (T[:-1] + T[1:]) / 10.0
            K = assemble_K(lam)
            b = M @ T
            b[0] += dt * 1.0
            T = np.linalg.solve(M + dt * K, b)
            if step in target_set:
                snapshots[step] = T.copy()
    elif mode == "nonlinear_root":
        for step in range(1, max_step + 1):
            b = M @ T
            b[0] += dt * 1.0
            b[-1] = 0.0

            def residual(T_next):
                c = (1.0 + 0.5 * (T_next[:-1] + T_next[1:]) / 10.0) / h
                KT = np.zeros(n)
                KT[0] = c[0] * (T_next[0] - T_next[1])
                KT[1:-1] = (
                    c[:-1] * (T_next[1:-1] - T_next[:-2])
                    + c[1:] * (T_next[1:-1] - T_next[2:])
                )
                return (M @ T_next) + dt * KT - b

            T = root(residual, T).x
            if step in target_set:
                snapshots[step] = T.copy()
    else:
        raise ValueError(f"Unknown FEM heat mode: {mode}")

    ordered_times = [s * dt for s in step_targets]
    ordered_profiles = [snapshots[s] for s in step_targets]
    return x_nodes, ordered_times, ordered_profiles


def plot_fem_heat_1d(
    mode="linear",
    L=10.0,
    n_elements=20,
    dt=0.1,
    t_values=None,
    default_t=20.0,
    title=None,
):
    if t_values is None:
        t_values = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]

    x_nodes, times, profiles = _fem_heat_1d_snapshots(mode, L, n_elements, dt, t_values)
    default_idx = int(np.argmin([abs(t - default_t) for t in times]))

    default_titles = {
        "linear": "1D Linear Heat Equation (FEM, λ = 1)",
        "nonlinear_lagged": "1D Nonlinear Heat Equation (Lagged λ = 1 + T/10)",
        "nonlinear_root": "1D Nonlinear Heat Equation (Root Finder, λ = 1 + T/10)",
    }
    base_title = title or default_titles.get(mode, "1D Heat Equation (FEM)")

    fig = go.Figure()
    for idx, (t_val, T_prof) in enumerate(zip(times, profiles)):
        fig.add_trace(
            go.Scatter(
                x=x_nodes,
                y=T_prof,
                mode="lines+markers",
                name=f"T(x, t = {t_val:g})",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=6),
                visible=(idx == default_idx),
            )
        )

    steps = []
    for idx, t_val in enumerate(times):
        vis = [i == idx for i in range(len(times))]
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis},
                    {"title": f"{base_title} — t = {t_val:g}"},
                ],
                label=f"{t_val:g}",
            )
        )

    fig.update_layout(
        title=f"{base_title} — t = {times[default_idx]:g}",
        xaxis_title="Position x",
        yaxis_title="Temperature T(x, t)",
        yaxis=dict(range=[-0.5, 10.5]),
        template="plotly_white",
        hovermode="x unified",
        height=480,
        sliders=[
            dict(
                active=default_idx,
                currentvalue={"prefix": "Final Time t = "},
                pad={"t": 45},
                steps=steps,
            )
        ],
    )
    return fig
