import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def forward_euler(f, t0, y0, t_end, h):
    t_points = np.arange(t0, t_end + h / 2, h)
    y_points = np.zeros_like(t_points, dtype=float)
    y_points[0] = y0

    for i in range(len(t_points) - 1):
        y_points[i + 1] = y_points[i] + h * f(t_points[i], y_points[i])

    return {
        "t_points": t_points,
        "y_points": y_points,
        "h": h,
        "method": "Forward Euler",
    }


def heun_method(f, t0, y0, t_end, h, iterations=1):
    t_points = np.arange(t0, t_end + h / 2, h)
    y_points = np.zeros_like(t_points, dtype=float)
    y_points[0] = y0
    cycle_history = []

    for i in range(len(t_points) - 1):
        t_i, y_i = t_points[i], y_points[i]
        k1 = f(t_i, y_i)
        y_pred = y_i + h * k1
        step_cycles = [y_pred]

        for _ in range(iterations):
            k2 = f(t_i + h, y_pred)
            y_pred = y_i + h * (k1 + k2) / 2.0
            step_cycles.append(y_pred)

        y_points[i + 1] = y_pred
        cycle_history.append(step_cycles)

    return {
        "t_points": t_points,
        "y_points": y_points,
        "cycle_history": cycle_history,
        "h": h,
        "iterations": iterations,
        "method": f"Heun's Method ({iterations} iter)",
    }


def midpoint_ode(f, t0, y0, t_end, h):
    t_points = np.arange(t0, t_end + h / 2, h)
    y_points = np.zeros_like(t_points, dtype=float)
    y_points[0] = y0

    for i in range(len(t_points) - 1):
        t_i, y_i = t_points[i], y_points[i]
        k1 = f(t_i, y_i)
        k2 = f(t_i + 0.5 * h, y_i + 0.5 * h * k1)
        y_points[i + 1] = y_points[i] + h * k2

    return {
        "t_points": t_points,
        "y_points": y_points,
        "h": h,
        "method": "Midpoint Method (RK2)",
    }


def ralston_method(f, t0, y0, t_end, h):
    t_points = np.arange(t0, t_end + h / 2, h)
    y_points = np.zeros_like(t_points, dtype=float)
    y_points[0] = y0

    for i in range(len(t_points) - 1):
        t_i, y_i = t_points[i], y_points[i]
        k1 = f(t_i, y_i)
        k2 = f(t_i + 0.75 * h, y_i + 0.75 * h * k1)
        y_points[i + 1] = y_points[i] + h * (k1 + 2.0 * k2) / 3.0

    return {
        "t_points": t_points,
        "y_points": y_points,
        "h": h,
        "method": "Ralston's Method (RK2)",
    }


def rk3_method(f, t0, y0, t_end, h):
    t_points = np.arange(t0, t_end + h / 2, h)
    y_points = np.zeros_like(t_points, dtype=float)
    y_points[0] = y0

    for i in range(len(t_points) - 1):
        t_i, y_i = t_points[i], y_points[i]
        k1 = f(t_i, y_i)
        k2 = f(t_i + 0.5 * h, y_i + 0.5 * h * k1)
        k3 = f(t_i + h, y_i - h * k1 + 2.0 * h * k2)
        y_points[i + 1] = y_points[i] + h * (k1 + 4.0 * k2 + k3) / 6.0

    return {
        "t_points": t_points,
        "y_points": y_points,
        "h": h,
        "method": "Classical RK3",
    }


def rk4_method(f, t0, y0, t_end, h):
    t_points = np.arange(t0, t_end + h / 2, h)
    y_points = np.zeros_like(t_points, dtype=float)
    y_points[0] = y0

    for i in range(len(t_points) - 1):
        t_i, y_i = t_points[i], y_points[i]
        k1 = f(t_i, y_i)
        k2 = f(t_i + 0.5 * h, y_i + 0.5 * h * k1)
        k3 = f(t_i + 0.5 * h, y_i + 0.5 * h * k2)
        k4 = f(t_i + h, y_i + h * k3)
        y_points[i + 1] = y_points[i] + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return {
        "t_points": t_points,
        "y_points": y_points,
        "h": h,
        "method": "Classical RK4",
    }


def plot_direction_field(
    f,
    t_range=(0, 4),
    y_range=(0, 10),
    y_true=None,
    trajectories=None,
    n_grid=16,
    title="Direction Field & Solution Trajectories",
):
    t_grid = np.linspace(t_range[0], t_range[1], n_grid)
    y_grid = np.linspace(y_range[0], y_range[1], n_grid)
    T, Y = np.meshgrid(t_grid, y_grid)
    slopes = f(T, Y)

    dt = (t_range[1] - t_range[0]) / (n_grid * 2.6)
    dy_scale = (y_range[1] - y_range[0]) / (t_range[1] - t_range[0])

    seg_t = []
    seg_y = []
    for i in range(n_grid):
        for j in range(n_grid):
            ti = T[i, j]
            yi = Y[i, j]
            s = slopes[i, j]
            norm = np.sqrt(1.0 + (s / dy_scale) ** 2)
            d_t = dt / norm
            d_y = s * d_t
            seg_t.extend([ti - d_t, ti + d_t, None])
            seg_y.extend([yi - d_y, yi + d_y, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=seg_t,
            y=seg_y,
            mode="lines",
            name="Slope Field f(t, y)",
            line=dict(color="rgba(148, 163, 184, 0.7)", width=1.5),
            hoverinfo="skip",
        )
    )

    t_dense = np.linspace(t_range[0], t_range[1], 300)
    if y_true is not None:
        fig.add_trace(
            go.Scatter(
                x=t_dense,
                y=y_true(t_dense),
                mode="lines",
                name="Exact Solution y(t)",
                line=dict(color="#111827", width=3, dash="dash"),
            )
        )

    palette = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED"]
    if trajectories:
        if isinstance(trajectories, dict) and "t_points" in trajectories:
            trajectories = {trajectories.get("method", "Approximation"): trajectories}
        for idx, (label, data) in enumerate(trajectories.items()):
            c = palette[idx % len(palette)]
            t_pts = data["t_points"] if isinstance(data, dict) else data[0]
            y_pts = data["y_points"] if isinstance(data, dict) else data[1]
            fig.add_trace(
                go.Scatter(
                    x=t_pts,
                    y=y_pts,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=c, width=2.2),
                    marker=dict(size=7, symbol="circle"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time t",
        yaxis_title="y(t)",
        xaxis=dict(range=list(t_range)),
        yaxis=dict(range=list(y_range)),
        template="plotly_white",
        hovermode="x unified",
        height=460,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
    )
    return fig


def plot_ode_comparison(
    t_range,
    y_true,
    results_dict,
    title="Explicit Runge-Kutta Methods vs. Exact Solution",
):
    t_dense = np.linspace(t_range[0], t_range[1], 300)
    y_exact_dense = y_true(t_dense)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Solution Trajectories y(t)", "Global Truncation Error |y_i - y(t_i)|"),
        row_heights=[0.62, 0.38],
    )

    fig.add_trace(
        go.Scatter(
            x=t_dense,
            y=y_exact_dense,
            mode="lines",
            name="Exact Solution",
            line=dict(color="#111827", width=3, dash="dash"),
        ),
        row=1,
        col=1,
    )

    palette = ["#2563EB", "#D97706", "#059669", "#DC2626", "#7C3AED", "#0891B2"]
    for idx, (label, res) in enumerate(results_dict.items()):
        c = palette[idx % len(palette)]
        t_pts = res["t_points"]
        y_pts = res["y_points"]
        err = np.maximum(np.abs(y_pts - y_true(t_pts)), 1e-16)

        fig.add_trace(
            go.Scatter(
                x=t_pts,
                y=y_pts,
                mode="lines+markers",
                name=label,
                line=dict(color=c, width=2),
                marker=dict(size=7),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_pts[1:],
                y=err[1:],
                mode="lines+markers",
                name=f"{label} Error",
                line=dict(color=c, width=1.8),
                marker=dict(size=6, symbol="diamond"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="y(t)", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Error", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Time t", row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    return fig


def plot_heun_iterations(
    f,
    y_true,
    t0=0.0,
    y0=2.0,
    t_end=4.0,
    h=1.0,
    max_iter=15,
    title="Heun's Predictor-Corrector: 1 vs. 15 Iterations",
):
    res_1 = heun_method(f, t0, y0, t_end, h, iterations=1)
    res_m = heun_method(f, t0, y0, t_end, h, iterations=max_iter)

    t_dense = np.linspace(t0, t_end, 300)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Global Trajectory Comparison",
            f"Corrector Convergence at Step 1 (t = {t0 + h})",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=t_dense,
            y=y_true(t_dense),
            mode="lines",
            name="Exact Solution",
            line=dict(color="#111827", width=3, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=res_1["t_points"],
            y=res_1["y_points"],
            mode="lines+markers",
            name="Heun (1 iteration)",
            line=dict(color="#2563EB", width=2.2),
            marker=dict(size=8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=res_m["t_points"],
            y=res_m["y_points"],
            mode="lines+markers",
            name=f"Heun ({max_iter} iterations)",
            line=dict(color="#DC2626", width=2.2),
            marker=dict(size=8, symbol="square"),
        ),
        row=1,
        col=1,
    )

    step1_cycles = res_m["cycle_history"][0]
    iters = list(range(len(step1_cycles)))
    true_step1 = float(y_true(t0 + h))

    fig.add_trace(
        go.Scatter(
            x=iters,
            y=step1_cycles,
            mode="lines+markers",
            name="Corrector Sequence y_1^(k)",
            line=dict(color="#7C3AED", width=2.2),
            marker=dict(size=7),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_iter],
            y=[true_step1, true_step1],
            mode="lines",
            name=f"Exact y({t0 + h}) = {true_step1:.4f}",
            line=dict(color="#111827", width=2, dash="dot"),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Time t", row=1, col=1)
    fig.update_yaxes(title_text="y(t)", row=1, col=1)
    fig.update_xaxes(title_text="Corrector Iteration k", row=1, col=2)
    fig.update_yaxes(title_text="Approximation y_1", row=1, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    return fig


def plot_rk4_stages(
    f,
    y_true,
    t0=0.0,
    y0=2.0,
    h=1.0,
    title="Geometric Anatomy of a Single Classical RK4 Step",
):
    k1 = f(t0, y0)
    t_half = t0 + 0.5 * h
    y_k1 = y0 + 0.5 * h * k1
    k2 = f(t_half, y_k1)
    y_k2 = y0 + 0.5 * h * k2
    k3 = f(t_half, y_k2)
    t1 = t0 + h
    y_k3 = y0 + h * k3
    k4 = f(t1, y_k3)
    y_rk4 = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    t_dense = np.linspace(t0, t1 + 0.15 * h, 200)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_dense,
            y=y_true(t_dense),
            mode="lines",
            name="Exact Solution y(t)",
            line=dict(color="#111827", width=3),
        )
    )

    # Trial step lines from (t0, y0)
    fig.add_trace(
        go.Scatter(
            x=[t0, t_half],
            y=[y0, y_k1],
            mode="lines+markers",
            name=f"Stage 1 → Midpoint (k1={k1:.2f})",
            line=dict(color="#2563EB", width=2, dash="dot"),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[t0, t_half],
            y=[y0, y_k2],
            mode="lines+markers",
            name=f"Stage 2 → Corrected Midpoint (k2={k2:.2f})",
            line=dict(color="#D97706", width=2, dash="dot"),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[t0, t1],
            y=[y0, y_k3],
            mode="lines+markers",
            name=f"Stage 3 → Endpoint Trial (k3={k3:.2f})",
            line=dict(color="#7C3AED", width=2, dash="dot"),
            marker=dict(size=8),
        )
    )

    # Short tangent slope segments at each stage evaluation point
    seg_len = 0.18 * h
    stages = [
        (t0, y0, k1, "k1", "#2563EB"),
        (t_half, y_k1, k2, "k2", "#D97706"),
        (t_half, y_k2, k3, "k3", "#7C3AED"),
        (t1, y_k3, k4, "k4", "#DC2626"),
    ]
    for ts, ys, ks, lbl, col in stages:
        fig.add_trace(
            go.Scatter(
                x=[ts, ts + seg_len],
                y=[ys, ys + ks * seg_len],
                mode="lines",
                name=f"Slope {lbl} = {ks:.2f}",
                line=dict(color=col, width=3.5),
                showlegend=(lbl == "k4"),
            )
        )

    # Final weighted RK4 step
    fig.add_trace(
        go.Scatter(
            x=[t0, t1],
            y=[y0, y_rk4],
            mode="lines+markers",
            name=f"Final RK4 Step (y1={y_rk4:.4f})",
            line=dict(color="#059669", width=3),
            marker=dict(size=11, symbol="star"),
        )
    )

    fig.add_vline(x=t_half, line_dash="dash", line_color="#9CA3AF", annotation_text="t0 + h/2")
    fig.add_vline(x=t1, line_dash="dash", line_color="#9CA3AF", annotation_text="t0 + h")

    fig.update_layout(
        title=title,
        xaxis_title="Time t",
        yaxis_title="y(t)",
        template="plotly_white",
        hovermode="closest",
        height=480,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
    )
    return fig


def rk4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def backward_euler(f, t0, y0, t_end, h):
    from scipy.optimize import root

    t_points = np.arange(t0, t_end + h / 2, h)
    y_points = np.zeros_like(t_points, dtype=float)
    y_points[0] = y0

    for i in range(len(t_points) - 1):
        t_next = t_points[i + 1]
        y_curr = y_points[i]
        sol = root(lambda y_val: y_val - (y_curr + h * f(t_next, y_val)), y_curr)
        y_points[i + 1] = sol.x[0]

    return {
        "t_points": t_points,
        "y_points": y_points,
        "h": h,
        "method": "Backward Euler",
    }


def plot_transient_function(
    t_range=(0.0, 10.0),
    n_coarse=12,
    title="Fast Transient vs. Slow Drift: y(t) = tanh((t - 5) / 0.1) - t",
):
    t_dense = np.linspace(t_range[0], t_range[1], 600)
    y_dense = np.tanh((t_dense - 5.0) / 0.1) - t_dense

    t_coarse = np.linspace(t_range[0], t_range[1], n_coarse)
    y_coarse = np.tanh((t_coarse - 5.0) / 0.1) - t_coarse

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_dense,
            y=y_dense,
            mode="lines",
            name="Exact Curve (Fine Resolution)",
            line=dict(color="#2563EB", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_coarse,
            y=y_coarse,
            mode="lines+markers",
            name=f"Coarse Uniform Step ({n_coarse} points)",
            line=dict(color="#DC2626", width=2, dash="dash"),
            marker=dict(size=8, symbol="circle"),
        )
    )

    fig.add_vrect(
        x0=4.7,
        x1=5.3,
        fillcolor="rgba(245, 158, 11, 0.18)",
        line_width=0,
        annotation_text="Sharp Transient",
        annotation_position="top left",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time t",
        yaxis_title="y(t)",
        template="plotly_white",
        hovermode="x unified",
        height=440,
        legend=dict(x=0.02, y=0.15, bgcolor="rgba(255, 255, 255, 0.85)"),
    )
    return fig


def plot_rk45_adaptive(
    sol,
    y_true=None,
    title="Adaptive Runge-Kutta-Fehlberg (RK45) Trajectory & Step Sizes",
):
    t_pts = sol.t
    y_pts = sol.y[0]
    step_sizes = np.diff(t_pts)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Solution Trajectory y(t) & Adaptive Nodes",
            "Adaptive Step Size h_i = t_{i+1} - t_i",
        ),
        row_heights=[0.62, 0.38],
    )

    if y_true is not None:
        t_dense = np.linspace(t_pts[0], t_pts[-1], 300)
        fig.add_trace(
            go.Scatter(
                x=t_dense,
                y=y_true(t_dense),
                mode="lines",
                name="Exact Solution",
                line=dict(color="#111827", width=3, dash="dash"),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=t_pts,
            y=y_pts,
            mode="lines+markers",
            name=f"RK45 Nodes ({len(t_pts)} steps)",
            line=dict(color="#2563EB", width=2.2),
            marker=dict(size=9, symbol="circle"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t_pts[:-1],
            y=step_sizes,
            mode="lines+markers",
            name="Step Size h",
            line=dict(color="#059669", width=2.2, shape="hv"),
            marker=dict(size=8, symbol="diamond"),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="y(t)", row=1, col=1)
    fig.update_yaxes(title_text="Step Size h", row=2, col=1)
    fig.update_xaxes(title_text="Time t", row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    return fig


def plot_euler_stability(
    t_end=20.0,
    y0=2.0,
    title="Numerical Stability on dy/dt = -y: Explicit vs. Implicit Euler",
):
    h_values = [0.2, 0.5, 0.8, 1.2, 1.8, 2.0, 2.15, 2.3]
    default_idx = 2  # h = 0.8

    t_dense = np.linspace(0.0, t_end, 400)
    y_exact = y0 * np.exp(-t_dense)

    fig = go.Figure()

    # Trace 0: Exact solution (always visible)
    fig.add_trace(
        go.Scatter(
            x=t_dense,
            y=y_exact,
            mode="lines",
            name="Exact Solution y(t) = 2e^(-t)",
            line=dict(color="#111827", width=3, dash="dash"),
            visible=True,
        )
    )

    for idx, h in enumerate(h_values):
        t_pts = np.arange(0.0, t_end + h / 2, h)
        y_exp = np.zeros_like(t_pts)
        y_imp = np.zeros_like(t_pts)
        y_exp[0] = y0
        y_imp[0] = y0
        for i in range(1, len(t_pts)):
            y_exp[i] = y_exp[i - 1] * (1.0 - h)
            y_imp[i] = y_imp[i - 1] / (1.0 + h)

        is_vis = idx == default_idx
        fig.add_trace(
            go.Scatter(
                x=t_pts,
                y=y_exp,
                mode="lines+markers",
                name=f"Forward Euler (h = {h})",
                line=dict(color="#2563EB", width=2.2),
                marker=dict(size=7, symbol="circle"),
                visible=is_vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_pts,
                y=y_imp,
                mode="lines+markers",
                name=f"Backward Euler (h = {h})",
                line=dict(color="#DC2626", width=2.2),
                marker=dict(size=7, symbol="x"),
                visible=is_vis,
            )
        )

    steps = []
    for idx, h in enumerate(h_values):
        vis = [True] + [False] * (2 * len(h_values))
        vis[1 + 2 * idx] = True
        vis[1 + 2 * idx + 1] = True
        regime = (
            "Monotonic decay"
            if h < 1.0
            else ("Damped oscillation" if h < 2.0 else ("Neutral oscillation" if h == 2.0 else "Unstable divergence!"))
        )
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis},
                    {"title": f"{title} — Step Size h = {h} ({regime})"},
                ],
                label=f"{h}",
            )
        )

    fig.update_layout(
        title=f"{title} — Step Size h = {h_values[default_idx]} (Monotonic decay)",
        xaxis_title="Time t",
        yaxis_title="y(t)",
        xaxis=dict(range=[0, t_end]),
        template="plotly_white",
        hovermode="x unified",
        height=520,
        legend=dict(x=0.58, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
        sliders=[
            dict(
                active=default_idx,
                currentvalue={"prefix": "Step size h = "},
                pad={"t": 45},
                steps=steps,
            )
        ],
    )
    return fig


def plot_stiff_ode(
    title="Analytical Solution of Stiff ODE: y(t) = 3 - 0.998e^(-1000t) - 2.002e^(-t)",
):
    limits = [0.005, 0.02, 0.1, 0.5, 1.5, 3.0, 5.0]
    default_idx = 5  # limit = 3.0

    fig = go.Figure()
    for idx, lim in enumerate(limits):
        t = np.linspace(0.0, lim, 500)
        y = 3.0 - 0.998 * np.exp(-1000.0 * t) - 2.002 * np.exp(-t)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                name=f"y(t) on [0, {lim}]",
                line=dict(color="#2563EB", width=3),
                visible=(idx == default_idx),
            )
        )

    steps = []
    for idx, lim in enumerate(limits):
        vis = [i == idx for i in range(len(limits))]
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": f"{title} (t_max = {lim})",
                        "xaxis.range": [0, lim],
                    },
                ],
                label=f"{lim}",
            )
        )

    fig.update_layout(
        title=f"{title} (t_max = {limits[default_idx]})",
        xaxis_title="Time t",
        yaxis_title="y(t)",
        xaxis=dict(range=[0, limits[default_idx]]),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        sliders=[
            dict(
                active=default_idx,
                currentvalue={"prefix": "Time Window t_max = "},
                pad={"t": 45},
                steps=steps,
            )
        ],
    )
    return fig


def plot_stiff_euler_comparison(
    t0=0.01,
    t_end=0.05,
    title="Stiff ODE on [0.01, 0.05]: Explicit vs. Implicit Euler",
):
    h_values = [0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
    default_idx = 1  # h = 0.0005

    def y_true(t):
        return 3.0 - 0.998 * np.exp(-1000.0 * t) - 2.002 * np.exp(-t)

    def f_stiff(t, y):
        return -1000.0 * y + 3000.0 - 2000.0 * np.exp(-t)

    t_dense = np.linspace(t0, t_end, 400)
    y0 = y_true(t0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_dense,
            y=y_true(t_dense),
            mode="lines",
            name="Analytical Solution",
            line=dict(color="#111827", width=3, dash="dash"),
            visible=True,
        )
    )

    for idx, h in enumerate(h_values):
        res_exp = forward_euler(f_stiff, t0, y0, t_end, h)
        res_imp = backward_euler(f_stiff, t0, y0, t_end, h)

        is_vis = idx == default_idx
        fig.add_trace(
            go.Scatter(
                x=res_exp["t_points"],
                y=res_exp["y_points"],
                mode="lines+markers",
                name=f"Explicit Euler (h = {h:.4f})",
                line=dict(color="#2563EB", width=2),
                marker=dict(size=6, symbol="circle"),
                visible=is_vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=res_imp["t_points"],
                y=res_imp["y_points"],
                mode="lines+markers",
                name=f"Implicit Euler (h = {h:.4f})",
                line=dict(color="#DC2626", width=2),
                marker=dict(size=6, symbol="x"),
                visible=is_vis,
            )
        )

    steps = []
    for idx, h in enumerate(h_values):
        vis = [True] + [False] * (2 * len(h_values))
        vis[1 + 2 * idx] = True
        vis[1 + 2 * idx + 1] = True
        status = "Stable" if h < 0.002 else "Unstable (h > 2/1000)!"
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis},
                    {"title": f"{title} — h = {h:.4f} ({status})"},
                ],
                label=f"{h:.4f}",
            )
        )

    fig.update_layout(
        title=f"{title} — h = {h_values[default_idx]:.4f} (Stable)",
        xaxis_title="Time t",
        yaxis_title="y(t)",
        xaxis=dict(range=[t0, t_end]),
        template="plotly_white",
        hovermode="x unified",
        height=520,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
        sliders=[
            dict(
                active=default_idx,
                currentvalue={"prefix": "Step size h = "},
                pad={"t": 45},
                steps=steps,
            )
        ],
    )
    return fig


def plot_pendulum_system(
    t,
    theta,
    theta_dot,
    title="Nonlinear Pendulum Motion via Reduction of Order (RK45)",
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Time Evolution: Angle & Angular Velocity",
            "Phase Portrait: θ'(t) vs. θ(t)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=theta,
            mode="lines+markers",
            name="Angle Θ(t) [rad]",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=theta_dot,
            mode="lines",
            name="Angular Velocity dΘ/dt [rad/s]",
            line=dict(color="#D97706", width=2, dash="dot"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=theta,
            y=theta_dot,
            mode="lines+markers",
            name="Phase Trajectory",
            line=dict(color="#059669", width=2.2),
            marker=dict(size=5),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Time t (s)", row=1, col=1)
    fig.update_yaxes(title_text="State Variables", row=1, col=1)
    fig.update_xaxes(title_text="Angle Θ (rad)", row=1, col=2)
    fig.update_yaxes(title_text="Angular Velocity dΘ/dt (rad/s)", row=1, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    return fig


# =====================================================================
# 3. Pedagogical PDE Solvers (1D Heat Equation)
#    Exposed in notebooks via {literalinclude} and exe.display_source()
# =====================================================================


def forward_euler_heat(T0, alpha, dx, dt, num_steps):
    T = T0.copy()
    T_history = [T.copy()]
    r = alpha * dt / dx**2
    for _ in range(num_steps):
        T[1:-1] = T[1:-1] + r * (T[:-2] - 2.0 * T[1:-1] + T[2:])
        T_history.append(T.copy())
    return np.array(T_history)


def backward_euler_heat(T0, alpha, dx, dt, num_steps):
    T = T0.copy()
    T_history = [T.copy()]
    n = len(T)
    r = alpha * dt / dx**2
    A = (
        np.diag((1.0 + 2.0 * r) * np.ones(n), 0)
        + np.diag(-r * np.ones(n - 1), -1)
        + np.diag(-r * np.ones(n - 1), 1)
    )
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    for _ in range(num_steps):
        b = T.copy()
        T = np.linalg.solve(A, b)
        T_history.append(T.copy())
    return np.array(T_history)


def crank_nicolson_heat(T0, alpha, dx, dt, num_steps):
    T = T0.copy()
    T_history = [T.copy()]
    n = len(T)
    r = alpha * dt / (2.0 * dx**2)
    A = (
        np.diag((1.0 + 2.0 * r) * np.ones(n), 0)
        + np.diag(-r * np.ones(n - 1), -1)
        + np.diag(-r * np.ones(n - 1), 1)
    )
    B = (
        np.diag((1.0 - 2.0 * r) * np.ones(n), 0)
        + np.diag(r * np.ones(n - 1), -1)
        + np.diag(r * np.ones(n - 1), 1)
    )
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    for _ in range(num_steps):
        b = B @ T
        b[0] = T[0]
        b[-1] = T[-1]
        T = np.linalg.solve(A, b)
        T_history.append(T.copy())
    return np.array(T_history)


crank_nicholson_heat = crank_nicolson_heat


def plot_forward_euler_heat(alpha=1.0, dx=0.05, dt_values=None, num_steps=30):
    if dt_values is None:
        dt_values = [0.0005, 0.0010, 0.00125, 0.00135, 0.0015]

    x = np.linspace(0.0, 1.0, int(round(1.0 / dx)) + 1)
    dx_actual = x[1] - x[0]
    T0 = np.sin(np.pi * x)

    fig = go.Figure()
    default_idx = 1

    # Add initial condition trace (always visible)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=T0,
            mode="lines",
            name="Initial T(x, 0) = sin(πx)",
            line=dict(color="#6B7280", width=2, dash="dash"),
            visible=True,
        )
    )

    traces_per_step = 3
    for idx, dt in enumerate(dt_values):
        vis = idx == default_idx
        T_hist = forward_euler_heat(T0, alpha, dx_actual, dt, num_steps)
        t_mid = (num_steps // 2) * dt
        t_final = num_steps * dt
        T_exact = np.exp(-alpha * (np.pi**2) * t_final) * np.sin(np.pi * x)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=T_exact,
                mode="lines",
                name=f"Exact Solution (t = {t_final:.4f})",
                line=dict(color="#1F2937", width=2.5),
                visible=vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=T_hist[num_steps // 2],
                mode="lines+markers",
                name=f"Forward Euler (step {num_steps // 2}, t = {t_mid:.4f})",
                line=dict(color="#3B82F6", width=2, dash="dot"),
                marker=dict(size=5),
                visible=vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=T_hist[-1],
                mode="lines+markers",
                name=f"Forward Euler (step {num_steps}, t = {t_final:.4f})",
                line=dict(color="#DC2626", width=2.5),
                marker=dict(size=6),
                visible=vis,
            )
        )

    steps = []
    for idx, dt in enumerate(dt_values):
        r_val = alpha * dt / (dx_actual**2)
        status = "Stable (r ≤ 0.5)" if r_val <= 0.5001 else "Unstable (r > 0.5)"
        vis_mask = [True] + [False] * (traces_per_step * len(dt_values))
        for k in range(traces_per_step):
            vis_mask[1 + idx * traces_per_step + k] = True
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis_mask},
                    {
                        "title": (
                            f"1D Heat Equation via Forward Euler "
                            f"(Δx = {dx_actual:.3f}, Δt = {dt:.5f}, r = {r_val:.2f} — {status})"
                        )
                    },
                ],
                label=f"{dt:.5f} (r={r_val:.2f})",
            )
        )

    r_def = alpha * dt_values[default_idx] / (dx_actual**2)
    fig.update_layout(
        title=(
            f"1D Heat Equation via Forward Euler "
            f"(Δx = {dx_actual:.3f}, Δt = {dt_values[default_idx]:.5f}, r = {r_def:.2f} — Stable (r ≤ 0.5))"
        ),
        xaxis_title="Position x",
        yaxis_title="Temperature T(x, t)",
        yaxis=dict(range=[-0.6, 1.4]),
        template="plotly_white",
        hovermode="x unified",
        height=520,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
        sliders=[
            dict(
                active=default_idx,
                currentvalue={"prefix": "Time step Δt = "},
                pad={"t": 45},
                steps=steps,
            )
        ],
    )
    return fig


def plot_backward_euler_heat(alpha=0.1, dx=0.05, dt_values=None, num_steps=25):
    if dt_values is None:
        dt_values = [0.005, 0.01, 0.025, 0.05, 0.1]

    x = np.linspace(0.0, 1.0, int(round(1.0 / dx)) + 1)
    dx_actual = x[1] - x[0]
    n = len(x)
    T0 = np.zeros_like(x)
    T0[n // 2] = 1.0

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=(
            "Backward Euler Temperature Profiles T(x, t)",
            "Tridiagonal Sparsity Pattern of Matrix A",
        ),
        horizontal_spacing=0.12,
    )

    sample_steps = [0, 2, 5, 12, num_steps]
    palette = ["#6B7280", "#2563EB", "#059669", "#D97706", "#DC2626"]
    default_idx = 2

    traces_per_dt = len(sample_steps)
    for idx, dt in enumerate(dt_values):
        vis = idx == default_idx
        T_hist = backward_euler_heat(T0, alpha, dx_actual, dt, num_steps)
        for s_idx, step_num in enumerate(sample_steps):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=T_hist[step_num],
                    mode="lines+markers" if step_num > 0 else "lines",
                    name=f"Step {step_num} (t = {step_num * dt:.3f})",
                    line=dict(
                        color=palette[s_idx],
                        width=2.2,
                        dash="dash" if step_num == 0 else "solid",
                    ),
                    marker=dict(size=5),
                    visible=vis,
                ),
                row=1,
                col=1,
            )

    # Sparsity pattern trace on col=2 (always visible)
    rows_idx, cols_idx = [], []
    for i in range(n):
        rows_idx.append(i)
        cols_idx.append(i)
        if i > 0 and i < n - 1:
            rows_idx.extend([i, i])
            cols_idx.extend([i - 1, i + 1])

    fig.add_trace(
        go.Scatter(
            x=cols_idx,
            y=rows_idx,
            mode="markers",
            name="Non-zero entries of A",
            marker=dict(symbol="square", size=7, color="#1E3A8A"),
            showlegend=False,
            visible=True,
        ),
        row=1,
        col=2,
    )

    steps = []
    total_temp_traces = traces_per_dt * len(dt_values)
    for idx, dt in enumerate(dt_values):
        r_val = alpha * dt / (dx_actual**2)
        vis_mask = [False] * total_temp_traces + [True]
        for k in range(traces_per_dt):
            vis_mask[idx * traces_per_dt + k] = True
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis_mask},
                    {
                        "title": (
                            f"Backward Euler 1D Heat Diffusion "
                            f"(Δt = {dt:.3f}, r = {r_val:.2f} — Unconditionally Stable)"
                        )
                    },
                ],
                label=f"Δt={dt:.3f} (r={r_val:.1f})",
            )
        )

    r_def = alpha * dt_values[default_idx] / (dx_actual**2)
    fig.update_xaxes(title_text="Position x", row=1, col=1)
    fig.update_yaxes(title_text="Temperature T(x, t)", range=[-0.05, 1.05], row=1, col=1)
    fig.update_xaxes(title_text="Column Index j", range=[-1, n], row=1, col=2)
    fig.update_yaxes(
        title_text="Row Index i",
        range=[n, -1],
        scaleanchor="x2",
        scaleratio=1,
        row=1,
        col=2,
    )
    fig.update_layout(
        title=(
            f"Backward Euler 1D Heat Diffusion "
            f"(Δt = {dt_values[default_idx]:.3f}, r = {r_def:.2f} — Unconditionally Stable)"
        ),
        template="plotly_white",
        hovermode="closest",
        height=520,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
        sliders=[
            dict(
                active=default_idx,
                currentvalue={"prefix": "Selected step: "},
                pad={"t": 45},
                steps=steps,
            )
        ],
    )
    return fig


def plot_pde_method_comparison(alpha=0.1, dx=0.1, dt=0.02, num_steps=6):
    x = np.linspace(0.0, 1.0, int(round(1.0 / dx)) + 1)
    dx_actual = x[1] - x[0]
    T0 = np.zeros_like(x)
    T0[len(x) // 2] = 1.0

    fe_sol = forward_euler_heat(T0, alpha, dx_actual, dt, num_steps)
    be_sol = backward_euler_heat(T0, alpha, dx_actual, dt, num_steps)
    cn_sol = crank_nicolson_heat(T0, alpha, dx_actual, dt, num_steps)

    fig = go.Figure()
    default_step = 1
    traces_per_step = 3

    for step_idx in range(num_steps + 1):
        vis = step_idx == default_step
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fe_sol[step_idx],
                mode="lines+markers",
                name="Forward Euler (1st-order time, explicit)",
                line=dict(color="#DC2626", width=2.2, dash="dot"),
                marker=dict(size=6, symbol="circle"),
                visible=vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=be_sol[step_idx],
                mode="lines+markers",
                name="Backward Euler (1st-order time, implicit)",
                line=dict(color="#2563EB", width=2.2, dash="dash"),
                marker=dict(size=6, symbol="square"),
                visible=vis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cn_sol[step_idx],
                mode="lines+markers",
                name="Crank-Nicolson (2nd-order time, implicit)",
                line=dict(color="#059669", width=2.6),
                marker=dict(size=7, symbol="diamond"),
                visible=vis,
            )
        )

    steps = []
    for step_idx in range(num_steps + 1):
        vis_mask = [False] * (traces_per_step * (num_steps + 1))
        for k in range(traces_per_step):
            vis_mask[step_idx * traces_per_step + k] = True
        t_val = step_idx * dt
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": vis_mask},
                    {
                        "title": (
                            f"Comparison of PDE Time-Stepping Schemes at Step {step_idx} "
                            f"(t = {t_val:.3f} s, r = {alpha * dt / dx_actual**2:.2f})"
                        )
                    },
                ],
                label=f"Step {step_idx} (t={t_val:.2f})",
            )
        )

    r_val = alpha * dt / dx_actual**2
    fig.update_layout(
        title=(
            f"Comparison of PDE Time-Stepping Schemes at Step {default_step} "
            f"(t = {default_step * dt:.3f} s, r = {r_val:.2f})"
        ),
        xaxis_title="Position x",
        yaxis_title="Temperature T(x, t)",
        yaxis=dict(range=[-0.1, 1.05]),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
        sliders=[
            dict(
                active=default_step,
                currentvalue={"prefix": "Time index: "},
                pad={"t": 45},
                steps=steps,
            )
        ],
    )
    return fig


