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
