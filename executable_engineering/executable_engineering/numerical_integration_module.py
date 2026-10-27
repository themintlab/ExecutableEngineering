import numpy as np
import plotly.graph_objects as go

def visual_left_riemann(f, a, b, h):

    x_left = np.arange(a, b, h)
    y_left = f(x_left)

    integral_approx = np.sum(y_left * h)

    x_span = b - a
    x_margin = x_span * 0.15
    plot_x_min = a - x_margin
    plot_x_max = b + x_margin

    x_vals = np.linspace(plot_x_min, plot_x_max, 400)
    y_vals = f(x_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Function f(x)',
        line=dict(color='lightgray', width=3)
    ))

    fig.add_trace(go.Bar(
        x=x_left,
        y=y_left,
        width=h,
        offset=0,
        name=f'Rectangles (h={h})',
        marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            line=dict(color='royalblue', width=1.5)
        ),
        hoverinfo='text',
        hovertext=[f"Interval: [{x:.3g}, {x+h:.3g}]<br>Height: {y:.4g}<br>Area: {y*h:.4g}"
                   for x, y in zip(x_left, y_left)]
    ))

    fig.add_trace(go.Scatter(
        x=x_left,
        y=y_left,
        mode='markers',
        name='Left Riemann Points',
        marker=dict(color='crimson', size=11, symbol='circle', line=dict(color='black', width=1)),
        hoverinfo='text',
        hovertext=[f"Left Point<br>x = {x:.4g}<br>f(x) = {y:.4g}" for x, y in zip(x_left, y_left)]
    ))

    fig.add_vline(x=a, line_width=2, line_dash="dash", line_color="black", annotation_text="a", annotation_position="top left")
    fig.add_vline(x=b, line_width=2, line_dash="dash", line_color="black", annotation_text="b", annotation_position="top right")

    fig.update_layout(
        title=f"Left Riemann Sum Approximation<br>Total Area Approximation ≈ {integral_approx:.5f}",
        xaxis_title="x",
        yaxis_title="f(x)",
        width=950,
        height=520,
        margin=dict(l=25, r=200, t=75, b=25),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eaeaea', zeroline=True, zerolinecolor='#cccccc'),
        yaxis=dict(showgrid=True, gridcolor='#eaeaea', zeroline=True, zerolinecolor='#cccccc'),
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255, 255, 255, 0.8)'),
    )

    return fig


def visual_right_riemann(f, a, b, h):

    x_right = np.arange(a + h, b + h, h)
    x_right = x_right[x_right <= b]
    y_right = f(x_right)

    integral_approx = np.sum(y_right * h)

    x_span = b - a
    x_margin = x_span * 0.15
    plot_x_min = a - x_margin
    plot_x_max = b + x_margin

    x_vals = np.linspace(plot_x_min, plot_x_max, 400)
    y_vals = f(x_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Function f(x)',
        line=dict(color='lightgray', width=3)
    ))

    fig.add_trace(go.Bar(
        x=x_right - h,
        y=y_right,
        width=h,
        offset=0,
        name=f'Rectangles (h={h})',
        marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            line=dict(color='royalblue', width=1.5)
        ),
        hoverinfo='text',
        hovertext=[
            f"Interval: [{x-h:.3g}, {x:.3g}]<br>"
            f"Height: {y:.4g}<br>"
            f"Area: {y*h:.4g}"
            for x, y in zip(x_right, y_right)
        ]
    ))

    fig.add_trace(go.Scatter(
        x=x_right,
        y=y_right,
        mode='markers',
        name='Right Riemann Points',
        marker=dict(
            color='crimson',
            size=11,
            symbol='circle',
            line=dict(color='black', width=1)
        ),
        hoverinfo='text',
        hovertext=[
            f"Right Point<br>x = {x:.4g}<br>f(x) = {y:.4g}"
            for x, y in zip(x_right, y_right)
        ]
    ))

    fig.add_vline(
        x=a, line_width=2, line_dash="dash",
        line_color="black", annotation_text="a",
        annotation_position="top left"
    )
    fig.add_vline(
        x=b, line_width=2, line_dash="dash",
        line_color="black", annotation_text="b",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"Right Riemann Sum Approximation<br>Total Area Approximation ≈ {integral_approx:.5f}",
        xaxis_title="x",
        yaxis_title="f(x)",
        width=950,
        height=520,
        margin=dict(l=25, r=200, t=75, b=25),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eaeaea',
                   zeroline=True, zerolinecolor='#cccccc'),
        yaxis=dict(showgrid=True, gridcolor='#eaeaea',
                   zeroline=True, zerolinecolor='#cccccc'),
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255, 255, 255, 0.8)')
    )

    return fig

def visual_midpoint_riemann(f, a, b, h):

    x_mid = np.arange(a + h/2, b, h)
    y_mid = f(x_mid)

    integral_approx = np.sum(y_mid * h)

    x_span = b - a
    x_margin = x_span * 0.15
    plot_x_min = a - x_margin
    plot_x_max = b + x_margin

    x_vals = np.linspace(plot_x_min, plot_x_max, 400)
    y_vals = f(x_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Function f(x)',
        line=dict(color='lightgray', width=3)
    ))

    fig.add_trace(go.Bar(
        x=x_mid,
        y=y_mid,
        width=h,
        offset=-h/2,
        name=f'Rectangles (h={h})',
        marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            line=dict(color='royalblue', width=1.5)
        ),
        hoverinfo='text',
        hovertext=[
            f"Interval: [{x-h/2:.3g}, {x+h/2:.3g}]<br>"
            f"Height: {y:.4g}<br>"
            f"Area: {y*h:.4g}"
            for x, y in zip(x_mid, y_mid)
        ]
    ))

    fig.add_trace(go.Scatter(
        x=x_mid,
        y=y_mid,
        mode='markers',
        name='Midpoint Riemann Points',
        marker=dict(
            color='crimson',
            size=11,
            symbol='circle',
            line=dict(color='black', width=1)
        ),
        hoverinfo='text',
        hovertext=[
            f"Midpoint<br>x = {x:.4g}<br>f(x) = {y:.4g}"
            for x, y in zip(x_mid, y_mid)
        ]
    ))

    fig.add_vline(
        x=a, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="a",
        annotation_position="top left"
    )

    fig.add_vline(
        x=b, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="b",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"Midpoint Riemann Sum Approximation<br>Total Area Approximation ≈ {integral_approx:.5f}",
        xaxis_title="x",
        yaxis_title="f(x)",
        width=950,
        height=520,
        margin=dict(l=25, r=200, t=75, b=25),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )

    return fig

def visual_trapezoid_rule(f, a, b, h):

    x_points = np.arange(a, b, h)
    x_points = np.append(x_points, b)

    x_points = np.unique(x_points)

    y_points = f(x_points)

    widths = np.diff(x_points)
    integral_approx = np.sum(
        widths * (y_points[:-1] + y_points[1:]) / 2
    )

    x_span = b - a
    x_margin = x_span * 0.15
    plot_x_min = a - x_margin
    plot_x_max = b + x_margin

    x_vals = np.linspace(plot_x_min, plot_x_max, 400)
    y_vals = f(x_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Function f(x)',
        line=dict(color='lightgray', width=3)
    ))

    for i in range(len(x_points) - 1):
        x_trap = [x_points[i], x_points[i],
                  x_points[i+1], x_points[i+1]]
        y_trap = [0, y_points[i],
                  y_points[i+1], 0]

        fig.add_trace(go.Scatter(
            x=x_trap,
            y=y_trap,
            mode='lines',
            fill='toself',
            fillcolor='rgba(135, 206, 250, 0.5)',
            line=dict(color='royalblue', width=1.5),
            showlegend=(i == 0),
            name=f'Trapezoids (h={h})',
            hoverinfo='text',
            text=[
                f"Interval: [{x_points[i]:.3g}, {x_points[i+1]:.3g}]<br>"
                f"Area ≈ {widths[i] * (y_points[i] + y_points[i+1]) / 2:.4g}"
            ] * 4
        ))

    fig.add_trace(go.Scatter(
        x=x_points,
        y=y_points,
        mode='markers',
        name='Trapezoid Rule Points',
        marker=dict(
            color='crimson',
            size=11,
            symbol='circle',
            line=dict(color='black', width=1)
        ),
        hoverinfo='text',
        hovertext=[
            f"Evaluation Point<br>x = {x:.4g}<br>f(x) = {y:.4g}"
            for x, y in zip(x_points, y_points)
        ]
    ))

    fig.add_vline(
        x=a, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="a",
        annotation_position="top left"
    )

    fig.add_vline(
        x=b, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="b",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"Trapezoidal Rule Approximation<br>Total Area Approximation ≈ {integral_approx:.5f}",
        xaxis_title="x",
        yaxis_title="f(x)",
        width=950,
        height=520,
        margin=dict(l=25, r=200, t=75, b=25),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )

    return fig


def visual_simpsons_13(f, a, b, h):

    n_float = (b - a) / h
    n = round(n_float)

    if not np.isclose(n_float, n) or n <= 0:
        raise ValueError("(b - a) must be a positive integer multiple of h.")

    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 Rule requires an even number of subintervals.")

    x_points = np.linspace(a, b, n + 1)
    y_points = np.asarray(f(x_points))

    integral_approx = (h / 3) * (
        y_points[0]
        + y_points[-1]
        + 4 * np.sum(y_points[1:-1:2])
        + 2 * np.sum(y_points[2:-1:2])
    )

    x_span = b - a
    x_margin = x_span * 0.15
    plot_x_min = a - x_margin
    plot_x_max = b + x_margin

    x_vals = np.linspace(plot_x_min, plot_x_max, 400)
    y_vals = f(x_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Function f(x)',
        line=dict(color='lightgray', width=3)
    ))

    for i in range(0, n, 2):

        x0, x1, x2 = x_points[i:i+3]
        y0, y1, y2 = y_points[i:i+3]

        x_curve = np.linspace(x0, x2, 100)

        L0 = ((x_curve - x1) * (x_curve - x2)) / ((x0 - x1) * (x0 - x2))
        L1 = ((x_curve - x0) * (x_curve - x2)) / ((x1 - x0) * (x1 - x2))
        L2 = ((x_curve - x0) * (x_curve - x1)) / ((x2 - x0) * (x2 - x1))

        y_curve = y0 * L0 + y1 * L1 + y2 * L2

        area = (h / 3) * (y0 + 4*y1 + y2)

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_curve, x_curve[::-1]]),
            y=np.concatenate([y_curve, np.zeros_like(y_curve)]),
            mode='lines',
            fill='toself',
            fillcolor='rgba(135, 206, 250, 0.5)',
            line=dict(color='royalblue', width=1.5),
            name="Parabolic Approximations",
            showlegend=(i == 0),
            hoverinfo='text',
            text=[
                f"Interval: [{x0:.3g}, {x2:.3g}]<br>"
                f"Simpson Area ≈ {area:.5g}"
            ] * (2 * len(x_curve))
        ))

        fig.add_trace(go.Scatter(
            x=x_curve,
            y=y_curve,
            mode='lines',
            line=dict(color='royalblue', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=x_points,
        y=y_points,
        mode='markers',
        name="Simpson's 1/3 Rule Points",
        marker=dict(
            color='crimson',
            size=11,
            symbol='circle',
            line=dict(color='black', width=1)
        ),
        hoverinfo='text',
        hovertext=[
            f"Evaluation Point<br>x = {x:.4g}<br>f(x) = {y:.4g}"
            for x, y in zip(x_points, y_points)
        ]
    ))

    fig.add_vline(
        x=a, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="a",
        annotation_position="top left"
    )

    fig.add_vline(
        x=b, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="b",
        annotation_position="top right"
    )

    fig.update_layout(
        title=(
            "Simpson's 1/3 Rule Approximation"
            f"<br>Total Area Approximation ≈ {integral_approx:.5f}"
        ),
        xaxis_title="x",
        yaxis_title="f(x)",
        width=950,
        height=520,
        margin=dict(l=25, r=200, t=75, b=25),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )

    return fig

def visual_simpsons_38(f, a, b, h):

    n_float = (b - a) / h
    n = round(n_float)

    if not np.isclose(n_float, n) or n <= 0:
        raise ValueError("(b - a) must be a positive integer multiple of h.")

    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 Rule requires a number of subintervals divisible by 3.")

    x_points = np.linspace(a, b, n + 1)
    y_points = np.asarray(f(x_points))

    weights = np.full(n + 1, 3.0)
    weights[0] = 1
    weights[-1] = 1
    weights[3:-1:3] = 2

    integral_approx = (3 * h / 8) * np.sum(weights * y_points)

    x_span = b - a
    x_margin = x_span * 0.15
    plot_x_min = a - x_margin
    plot_x_max = b + x_margin

    x_vals = np.linspace(plot_x_min, plot_x_max, 400)
    y_vals = f(x_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Function f(x)',
        line=dict(color='lightgray', width=3)
    ))

    for i in range(0, n, 3):

        x_group = x_points[i:i+4]
        y_group = y_points[i:i+4]

        x_curve = np.linspace(x_group[0], x_group[-1], 100)

        y_curve = np.zeros_like(x_curve)

        for j in range(4):
            basis = np.ones_like(x_curve)

            for k in range(4):
                if k != j:
                    basis *= (
                        (x_curve - x_group[k])
                        / (x_group[j] - x_group[k])
                    )

            y_curve += y_group[j] * basis

        area = (3 * h / 8) * (
            y_group[0] + 3*y_group[1]
            + 3*y_group[2] + y_group[3]
        )

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_curve, x_curve[::-1]]),
            y=np.concatenate([y_curve, np.zeros_like(y_curve)]),
            mode='lines',
            fill='toself',
            fillcolor='rgba(135, 206, 250, 0.5)',
            line=dict(color='royalblue', width=1.5),
            name='Cubic Approximations',
            showlegend=(i == 0),
            hoverinfo='text',
            text=[
                f"Interval: [{x_group[0]:.3g}, {x_group[-1]:.3g}]<br>"
                f"Simpson 3/8 Area ≈ {area:.5g}"
            ] * (2 * len(x_curve))
        ))

        fig.add_trace(go.Scatter(
            x=x_curve,
            y=y_curve,
            mode='lines',
            line=dict(color='royalblue', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=x_points,
        y=y_points,
        mode='markers',
        name="Simpson's 3/8 Rule Points",
        marker=dict(
            color='crimson',
            size=11,
            symbol='circle',
            line=dict(color='black', width=1)
        ),
        hoverinfo='text',
        hovertext=[
            f"Evaluation Point<br>x = {x:.4g}<br>f(x) = {y:.4g}"
            for x, y in zip(x_points, y_points)
        ]
    ))

    fig.add_vline(
        x=a, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="a",
        annotation_position="top left"
    )

    fig.add_vline(
        x=b, line_width=2, line_dash="dash",
        line_color="black",
        annotation_text="b",
        annotation_position="top right"
    )

    fig.update_layout(
        title=(
            "Simpson's 3/8 Rule Approximation"
            f"<br>Total Area Approximation ≈ {integral_approx:.5f}"
        ),
        xaxis_title="x",
        yaxis_title="f(x)",
        width=950,
        height=520,
        margin=dict(l=25, r=200, t=75, b=25),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#eaeaea',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )

    return fig
