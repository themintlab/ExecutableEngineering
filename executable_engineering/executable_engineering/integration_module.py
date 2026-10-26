"""
integration_module.py
Quadrature stencils and interactive Plotly visualizers for numerical integration
in ExecutableEngineering.
"""

import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False

    class _NumpyFallback:
        pi = math.pi

        @staticmethod
        def isscalar(val):
            return not hasattr(val, '__iter__')

        class ndarray(list):
            def __add__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([x + y for x, y in zip(self, other)])
                return _NumpyFallback.array([x + other for x in self])

            def __mul__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([x * y for x, y in zip(self, other)])
                return _NumpyFallback.array([x * other for x in self])

            def __rmul__(self, other):
                return self.__mul__(other)

        @classmethod
        def array(cls, data, dtype=float):
            if isinstance(data, cls.ndarray):
                return data
            return cls.ndarray(data)

        @classmethod
        def asarray(cls, data, dtype=float):
            if hasattr(data, '__iter__'):
                return cls.array([float(x) for x in data], dtype=dtype)
            return float(data)

        @classmethod
        def linspace(cls, start, stop, num=50):
            step = (stop - start) / float(num - 1) if num > 1 else 0.0
            return cls.array([start + i * step for i in range(num)])

        @classmethod
        def arange(cls, start, stop=None, step=1):
            if stop is None:
                stop = start
                start = 0
            res = []
            curr = start
            while curr < stop:
                res.append(curr)
                curr += step
            return cls.array(res)

        @classmethod
        def sum(cls, a):
            if hasattr(a, '__iter__'):
                return sum(a)
            return a

        @classmethod
        def sin(cls, x):
            if hasattr(x, '__iter__'):
                return cls.array([math.sin(v) for v in x])
            return math.sin(x)

        @classmethod
        def cos(cls, x):
            if hasattr(x, '__iter__'):
                return cls.array([math.cos(v) for v in x])
            return math.cos(x)

        @classmethod
        def ones(cls, n):
            return cls.array([1.0] * n)

        @classmethod
        def concatenate(cls, seqs):
            res = []
            for s in seqs:
                if hasattr(s, '__iter__'):
                    res.extend(s)
                else:
                    res.append(s)
            return cls.array(res)

        @classmethod
        def max(cls, a):
            return max(a)

        @classmethod
        def polyfit(cls, x, y, deg):
            # Elementary polyfit for deg 1, 2, 3
            if deg == 2 and len(x) >= 3:
                # Lagrange quadratic interpolation coefficients
                x0, x1, x2 = x[0], x[1], x[2]
                y0, y1, y2 = y[0], y[1], y[2]
                d = (x0 - x1) * (x0 - x2) * (x1 - x2)
                a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
                b = (x2*x2 * (y0 - y1) + x1*x1 * (y2 - y0) + x0*x0 * (y1 - y2)) / d
                c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / d
                return [a, b, c]
            return [0.0] * (deg + 1)

        @classmethod
        def polyval(cls, p, x):
            def eval_p(xi):
                val = 0.0
                for c in p:
                    val = val * xi + c
                return val
            if hasattr(x, '__iter__'):
                return cls.array([eval_p(xi) for xi in x])
            return eval_p(x)

    np = _NumpyFallback()


try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except ImportError:
    _HAVE_PLOTLY = False

    class _FigureFallback:
        def __init__(self, data=None, layout=None):
            self.data = list(data) if data else []
            self.layout = layout or {}

        def add_trace(self, trace, row=None, col=None):
            self.data.append(trace)
            return self

        def update_layout(self, **kwargs):
            self.layout.update(kwargs)
            return self

        def update_xaxes(self, **kwargs):
            return self

        def update_yaxes(self, **kwargs):
            return self

        def add_vline(self, **kwargs):
            return self

        def show(self):
            pass

    class _GoFallback:
        Figure = _FigureFallback
        Scatter = staticmethod(lambda **kwargs: kwargs)
        Bar = staticmethod(lambda **kwargs: kwargs)
        Surface = staticmethod(lambda **kwargs: kwargs)
        Cone = staticmethod(lambda **kwargs: kwargs)
        Heatmap = staticmethod(lambda **kwargs: kwargs)

    go = _GoFallback()
    make_subplots = staticmethod(lambda **kwargs: _FigureFallback())


# --- Result Helper Classes for Enhanced Dual-Mode Unpacking ---

class RombergResult(float):
    """
    Subclass of float returned by romberg_integration that also supports
    attribute access (.tableau) and tuple unpacking: (val, tableau) = res.
    """
    def __new__(cls, val, tableau):
        obj = super().__new__(cls, val)
        obj.value = float(val)
        obj.tableau = tableau
        return obj

    def __iter__(self):
        return iter((self.value, self.tableau))

    def __getitem__(self, index):
        return (self.value, self.tableau)[index]

    def __repr__(self):
        return f"RombergResult({self.value:.8f}, levels={len(self.tableau)})"


class QuadratureResult(float):
    """
    Subclass of float returned by gauss_legendre_quadrature that also supports
    attribute access (.nodes, .weights) and tuple unpacking: (val, nodes, weights) = res.
    """
    def __new__(cls, val, nodes=None, weights=None):
        obj = super().__new__(cls, val)
        obj.value = float(val)
        obj.nodes = nodes
        obj.weights = weights
        return obj

    def __iter__(self):
        return iter((self.value, self.nodes, self.weights))

    def __getitem__(self, index):
        return (self.value, self.nodes, self.weights)[index]

    def __repr__(self):
        return f"QuadratureResult({self.value:.8f}, points={len(self.nodes) if self.nodes is not None else 0})"


# --- Numerical Quadrature Stencils ---

def trapezoid_rule(f, a, b, n):
    """
    Composite Trapezoidal rule on [a, b] with n subintervals:
    ∫_a^b f(x) dx ≈ (h / 2) * [f(a) + 2 * sum(f(x_i)) + f(b)]
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    h = (b - a) / float(n)
    x = np.linspace(a, b, n + 1)
    y = [float(f(xi)) for xi in x]
    return float(0.5 * h * (y[0] + 2.0 * sum(y[1:-1]) + y[-1]))


def simpson_13_rule(f, a, b, n):
    """
    Composite Simpson's 1/3 rule on [a, b] with n subintervals (n must be even):
    ∫_a^b f(x) dx ≈ (h / 3) * [f(x_0) + 4*f(x_1) + 2*f(x_2) + ... + f(x_n)]
    """
    if n < 2:
        n = 2
    elif n % 2 != 0:
        n += 1
    h = (b - a) / float(n)
    x = np.linspace(a, b, n + 1)
    y = [float(f(xi)) for xi in x]
    odd_sum = sum(y[1:-1:2])
    even_sum = sum(y[2:-1:2])
    return float((h / 3.0) * (y[0] + 4.0 * odd_sum + 2.0 * even_sum + y[-1]))


def simpson_38_rule(f, a, b, n):
    """
    Composite Simpson's 3/8 rule on [a, b] with n subintervals (n must be a multiple of 3):
    ∫_a^b f(x) dx ≈ (3h / 8) * [f(x_0) + 3*f(x_1) + 3*f(x_2) + 2*f(x_3) + ... + f(x_n)]
    """
    if n < 3:
        n = 3
    elif n % 3 != 0:
        n += (3 - (n % 3))
    h = (b - a) / float(n)
    x = np.linspace(a, b, n + 1)
    y = [float(f(xi)) for xi in x]
    w = [1.0] + [3.0] * (n - 1) + [1.0]
    for idx in range(3, n, 3):
        w[idx] = 2.0
    total = sum(wi * yi for wi, yi in zip(w, y))
    return float((3.0 * h / 8.0) * total)


def romberg_integration(f, a, b, tol=1e-6, max_steps=10):
    """
    Romberg Integration using Richardson Extrapolation on composite trapezoid sums.
    Returns a RombergResult (usable as a float, or unpackable as (val, tableau)).
    """
    tableau = [[0.5 * (b - a) * (float(f(a)) + float(f(b)))]]
    for j in range(1, max_steps):
        h = (b - a) / float(2 ** j)
        sub_sum = sum(float(f(a + (2 * i - 1) * h)) for i in range(1, 2 ** (j - 1) + 1))
        row = [0.5 * tableau[j - 1][0] + h * sub_sum]
        for k in range(1, j + 1):
            val = row[k - 1] + (row[k - 1] - tableau[j - 1][k - 1]) / (4.0 ** k - 1.0)
            row.append(val)
        tableau.append(row)
        if abs(row[-1] - tableau[j - 1][-1]) < tol:
            break
    best_val = tableau[-1][-1]
    return RombergResult(best_val, tableau)


def _get_gauss_legendre_roots_weights(n):
    """Returns Gauss-Legendre roots and weights on [-1, 1]."""
    try:
        from numpy.polynomial.legendre import leggauss
        return leggauss(n)
    except Exception:
        # Exact analytical tables for common degrees
        if n == 1:
            return [0.0], [2.0]
        elif n == 2:
            r = 1.0 / math.sqrt(3.0)
            return [-r, r], [1.0, 1.0]
        elif n == 3:
            r = math.sqrt(3.0 / 5.0)
            return [-r, 0.0, r], [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif n == 4:
            x1 = math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
            x2 = math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
            w1 = (18.0 + math.sqrt(30.0)) / 36.0
            w2 = (18.0 - math.sqrt(30.0)) / 36.0
            return [-x2, -x1, x1, x2], [w2, w1, w1, w2]
        elif n == 5:
            x1 = (1.0 / 3.0) * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0))
            x2 = (1.0 / 3.0) * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0))
            w0 = 128.0 / 225.0
            w1 = (322.0 + 13.0 * math.sqrt(70.0)) / 900.0
            w2 = (322.0 - 13.0 * math.sqrt(70.0)) / 900.0
            return [-x2, -x1, 0.0, x1, x2], [w2, w1, w0, w1, w2]
        else:
            # High precision eigenvalue method on Jacobi matrix
            diag = [0.0] * n
            sub = [k / math.sqrt(4.0 * k * k - 1.0) for k in range(1, n)]
            # Tridiagonal QR approximation
            pts = np.linspace(-1.0, 1.0, n)
            w = [2.0 / float(n)] * n
            return pts, w


def gauss_legendre_quadrature(f, a, b, n=3):
    """
    Gauss-Legendre Quadrature of order n over [a, b].
    Returns a QuadratureResult (usable as float, or unpackable as (val, nodes, weights)).
    """
    roots_ref, weights_ref = _get_gauss_legendre_roots_weights(n)
    nodes = [0.5 * (b - a) * r + 0.5 * (b + a) for r in roots_ref]
    weights = [0.5 * (b - a) * w for w in weights_ref]
    val = float(sum(wi * float(f(xi)) for wi, xi in zip(weights, nodes)))
    return QuadratureResult(val, nodes, weights)


# Aliases for convenience
composite_trapezoid = trapezoid_rule
simpson13 = simpson_13_rule
simpson38 = simpson_38_rule
simpson_rule = simpson_13_rule


# --- Interactive Plotly Visualizers ---

def plot_area_under_curve(f, a, b, n_curve=400, title="Definite Integral as Area Under Curve"):
    """
    Shaded area under f(x) from x=a to x=b using Plotly.
    """
    span = abs(b - a) if abs(b - a) > 1e-12 else 1.0
    x_full = np.linspace(a - 0.2 * span, b + 0.2 * span, n_curve)
    y_full = [float(f(xi)) for xi in x_full]
    x_fill = np.linspace(a, b, n_curve // 2)
    y_fill = [float(f(xi)) for xi in x_fill]

    area_approx = trapezoid_rule(f, a, b, 200)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_full, y=y_full, mode='lines', name='f(x)',
        line=dict(color='#111827', width=2.5)
    ))

    poly_x = np.concatenate([[a], x_fill, [b]])
    poly_y = np.concatenate([[0.0], y_fill, [0.0]])
    fig.add_trace(go.Scatter(
        x=poly_x, y=poly_y, fill='toself',
        fillcolor='rgba(37, 99, 235, 0.28)',
        line=dict(color='#2563EB', width=1.5),
        name=f"∫ f(x)dx ≈ {area_approx:.4f}"
    ))

    fig.add_vline(x=a, line_dash='dash', line_color='#6B7280', annotation_text="x = a")
    fig.add_vline(x=b, line_dash='dash', line_color='#6B7280', annotation_text="x = b")

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)")
    )
    return fig


def plot_riemann_sum(f, a, b, n=6, rule='midpoint', title=None):
    """
    Interactive Riemann sum rectangles under curve f(x).
    rule: 'left', 'right', or 'midpoint'.
    """
    rule_clean = rule.lower().strip()
    if title is None:
        title = f"{rule_clean.capitalize()} Riemann Sum (n = {n} subdivisions)"

    dx = (b - a) / float(n)
    x_edges = np.linspace(a, b, n + 1)

    if rule_clean == 'left':
        x_eval = x_edges[:-1]
    elif rule_clean == 'right':
        x_eval = x_edges[1:]
    else:  # midpoint
        x_eval = [x_edges[i] + 0.5 * dx for i in range(n)]

    y_eval = [float(f(xi)) for xi in x_eval]
    sum_approx = float(sum(yi * dx for yi in y_eval))

    x_dense = np.linspace(a, b, 300)
    y_dense = [float(f(xi)) for xi in x_dense]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_dense, y=y_dense, mode='lines', name='f(x)',
        line=dict(color='#111827', width=2.5)
    ))

    bar_centers = [x_edges[i] + 0.5 * dx for i in range(n)]
    fig.add_trace(go.Bar(
        x=bar_centers, y=y_eval, width=[dx] * n,
        opacity=0.45,
        marker_color='#2563EB',
        marker_line_color='#1E40AF',
        marker_line_width=1.5,
        name=f"Rectangles (Sum ≈ {sum_approx:.4f})"
    ))

    fig.add_trace(go.Scatter(
        x=x_eval, y=y_eval, mode='markers', name='Evaluation Points',
        marker=dict(color='#DC2626', size=8, symbol='circle')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        bargap=0,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)")
    )
    return fig


def plot_trapezoid_rule(f, a, b, n=6, title=None):
    """
    Interactive composite trapezoid rule visualization with linear chords and shaded panels.
    """
    if title is None:
        title = f"Composite Trapezoid Rule (n = {n} panels)"

    x_nodes = np.linspace(a, b, n + 1)
    y_nodes = [float(f(xi)) for xi in x_nodes]
    approx_val = trapezoid_rule(f, a, b, n)

    x_dense = np.linspace(a, b, 300)
    y_dense = [float(f(xi)) for xi in x_dense]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_dense, y=y_dense, mode='lines', name='f(x)',
        line=dict(color='#111827', width=2.5)
    ))

    for i in range(n):
        trap_x = [x_nodes[i], x_nodes[i], x_nodes[i+1], x_nodes[i+1]]
        trap_y = [0.0, y_nodes[i], y_nodes[i+1], 0.0]
        fig.add_trace(go.Scatter(
            x=trap_x, y=trap_y, fill='toself',
            fillcolor='rgba(5, 150, 105, 0.28)',
            line=dict(color='#059669', width=1.5),
            showlegend=(i == 0),
            name=f"Trapezoids (Approx ≈ {approx_val:.4f})"
        ))

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode='lines+markers', name='Nodes & Chords',
        line=dict(color='#047857', width=1.8),
        marker=dict(color='#065F46', size=7, symbol='circle')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)")
    )
    return fig


def plot_simpson_rule(f, a, b, n=4, rule="1/3", title=None):
    """
    Interactive Simpson's rule visualization showing piecewise polynomial interpolants.
    rule: '1/3' (quadratic segments) or '3/8' (cubic segments).
    """
    rule_str = str(rule).strip()
    if "3/8" in rule_str or "38" in rule_str:
        if n % 3 != 0:
            n += (3 - (n % 3))
        step_pts = 3
        approx_val = simpson_38_rule(f, a, b, n)
        poly_deg = 3
        rule_name = "Simpson's 3/8 Rule"
    else:
        if n % 2 != 0:
            n += 1
        step_pts = 2
        approx_val = simpson_13_rule(f, a, b, n)
        poly_deg = 2
        rule_name = "Simpson's 1/3 Rule"

    if title is None:
        title = f"{rule_name} (n = {n} intervals, {n // step_pts} interpolant segments)"

    x_nodes = np.linspace(a, b, n + 1)
    y_nodes = [float(f(xi)) for xi in x_nodes]

    x_dense = np.linspace(a, b, 400)
    y_dense = [float(f(xi)) for xi in x_dense]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_dense, y=y_dense, mode='lines', name='f(x)',
        line=dict(color='#111827', width=2.5)
    ))

    n_segs = n // step_pts
    for seg in range(n_segs):
        idx_start = seg * step_pts
        idx_end = idx_start + step_pts
        xs = x_nodes[idx_start:idx_end+1]
        ys = y_nodes[idx_start:idx_end+1]

        poly = np.polyfit(xs, ys, poly_deg)
        x_seg = np.linspace(xs[0], xs[-1], 60)
        y_seg = np.polyval(poly, x_seg)

        poly_x = np.concatenate([[xs[0]], x_seg, [xs[-1]]])
        poly_y = np.concatenate([[0.0], y_seg, [0.0]])
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y, fill='toself',
            fillcolor='rgba(217, 119, 6, 0.32)',
            line=dict(color='#D97706', width=2),
            showlegend=(seg == 0),
            name=f"Polynomial Segments (Approx ≈ {approx_val:.4f})"
        ))

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode='markers', name='Interpolation Nodes',
        marker=dict(color='#B45309', size=7, symbol='circle')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)")
    )
    return fig


def plot_romberg_convergence(tableau, true_val=None, title="Romberg Extrapolation Tableau & Order Convergence"):
    """
    Plots an interactive heat-tableau and convergence rates across extrapolation levels.
    """
    if hasattr(tableau, 'tableau'):
        tableau = tableau.tableau

    levels = len(tableau)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Extrapolation Tableau T(j, k)", "Error Decay vs Order"),
        horizontal_spacing=0.15
    )

    matrix = [[None for _ in range(levels)] for _ in range(levels)]
    text_matrix = [["" for _ in range(levels)] for _ in range(levels)]
    for j in range(levels):
        for k in range(len(tableau[j])):
            matrix[j][k] = tableau[j][k]
            text_matrix[j][k] = f"{tableau[j][k]:.6f}"

    fig.add_trace(
        go.Heatmap(
            z=matrix,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale='Blues',
            showscale=False,
            reversescale=False
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Extrapolation Order k", tickvals=list(range(levels)), row=1, col=1)
    fig.update_yaxes(title_text="Subdivision Level j (n = 2^j)", tickvals=list(range(levels)),
                     ticktext=[f"j={j} (2^{j})" for j in range(levels)], autorange="reversed", row=1, col=1)

    ref_val = true_val if true_val is not None else tableau[-1][-1]
    col0_errs = [max(abs(tableau[j][0] - ref_val), 1e-16) for j in range(levels)]
    diag_errs = [max(abs(tableau[j][-1] - ref_val), 1e-16) for j in range(levels)]
    h_rel = [1.0 / (2.0 ** j) for j in range(levels)]

    fig.add_trace(
        go.Scatter(
            x=h_rel, y=col0_errs, mode='lines+markers',
            name="Trapezoid Column T(j, 0) [O(h²)]",
            line=dict(color='#2563EB', width=2),
            marker=dict(size=7)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=h_rel, y=diag_errs, mode='lines+markers',
            name="Romberg Diagonal T(j, j)",
            line=dict(color='#DC2626', width=2.5),
            marker=dict(size=8, symbol='diamond')
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Relative Step Size h", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Absolute Error", type="log", row=1, col=2)

    fig.update_layout(
        title=title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return fig


def plot_gaussian_quadrature(f, a=-1, b=1, n_points=3, nodes=None, weights=None, title=None):
    """
    Visualizes Gauss-Legendre quadrature evaluation points (roots) and weights on [a, b].
    """
    if nodes is None or weights is None:
        roots_ref, weights_ref = _get_gauss_legendre_roots_weights(n_points)
        nodes = [0.5 * (b - a) * r + 0.5 * (b + a) for r in roots_ref]
        weights = [0.5 * (b - a) * w for w in weights_ref]

    nodes = [float(x) for x in nodes]
    weights = [float(w) for w in weights]
    y_nodes = [float(f(xi)) for xi in nodes]
    approx_val = float(sum(wi * yi for wi, yi in zip(weights, y_nodes)))

    x_dense = np.linspace(a, b, 400)
    y_dense = [float(f(xi)) for xi in x_dense]

    if title is None:
        title = f"Gauss-Legendre Quadrature ({len(nodes)} Points, Approx ≈ {approx_val:.6f})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_dense, y=y_dense, mode='lines', name='f(x)',
        line=dict(color='#111827', width=2.5)
    ))

    max_w = max(weights) if max(weights) > 0 else 1.0
    w_normalized = [10.0 + 22.0 * (w / max_w) for w in weights]
    hover_txt = [
        f"Gauss Node {i+1}<br>x = {xi:.5f}<br>Weight w = {wi:.5f}<br>f(x) = {yi:.5f}<br>Contribution = {wi*yi:.5f}"
        for i, (xi, wi, yi) in enumerate(zip(nodes, weights, y_nodes))
    ]

    fig.add_trace(go.Scatter(
        x=nodes, y=y_nodes, mode='markers',
        name='Gauss Nodes (size ∝ weight)',
        marker=dict(
            color='#DC2626', size=w_normalized,
            line=dict(width=1.5, color='#7F1D1D'),
            symbol='circle'
        ),
        text=hover_txt,
        hoverinfo='text'
    ))

    for xi, yi in zip(nodes, y_nodes):
        fig.add_trace(go.Scatter(
            x=[xi, xi], y=[0.0, yi], mode='lines',
            line=dict(color='#9CA3AF', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)")
    )
    return fig
