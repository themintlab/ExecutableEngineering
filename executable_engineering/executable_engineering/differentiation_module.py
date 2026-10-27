"""
differentiation_module.py
Analytical models, finite difference stencils, and interactive Plotly visualizers
for numerical differentiation in ExecutableEngineering.
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
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    row_slice, col_slice = key
                    sub_rows = super().__getitem__(row_slice)
                    if isinstance(sub_rows, list):
                        return _NumpyFallback.array([_NumpyFallback.array(r)[col_slice] for r in sub_rows])
                    return sub_rows[col_slice]
                res = super().__getitem__(key)
                if isinstance(res, list):
                    return _NumpyFallback.array(res)
                return res

            def __add__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([x + y for x, y in zip(self, other)])
                return _NumpyFallback.array([x + other for x in self])

            def __radd__(self, other):
                return self.__add__(other)

            def __sub__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([x - y for x, y in zip(self, other)])
                return _NumpyFallback.array([x - other for x in self])

            def __rsub__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([y - x for x, y in zip(self, other)])
                return _NumpyFallback.array([other - x for x in self])

            def __mul__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([x * y for x, y in zip(self, other)])
                return _NumpyFallback.array([x * other for x in self])

            def __rmul__(self, other):
                return self.__mul__(other)

            def __truediv__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([x / y for x, y in zip(self, other)])
                return _NumpyFallback.array([x / other for x in self])

            def __rtruediv__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([y / x for x, y in zip(self, other)])
                return _NumpyFallback.array([other / x for x in self])

            def __pow__(self, other):
                return _NumpyFallback.array([x ** other for x in self])

            def __neg__(self):
                return _NumpyFallback.array([-x for x in self])

            def __gt__(self, other):
                if hasattr(other, '__iter__'):
                    return _NumpyFallback.array([x > y for x, y in zip(self, other)])
                return _NumpyFallback.array([x > other for x in self])

            def flatten(self):
                flat = []
                for item in self:
                    if hasattr(item, '__iter__'):
                        flat.extend(_NumpyFallback.array(item).flatten())
                    else:
                        flat.append(item)
                return _NumpyFallback.array(flat)

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
        def log(cls, x):
            if hasattr(x, '__iter__'):
                return cls.array([math.log(v) for v in x])
            return math.log(x)

        @classmethod
        def abs(cls, x):
            if hasattr(x, '__iter__'):
                return cls.array([math.fabs(v) for v in x])
            return math.fabs(x)

        @classmethod
        def where(cls, cond, a, b):
            if hasattr(cond, '__iter__'):
                return cls.array([av if cv else bv for cv, av, bv in zip(cond, a, b)])
            return a if cond else b

        @classmethod
        def meshgrid(cls, x, y):
            X = cls.array([list(x) for _ in y])
            Y = cls.array([[yj for _ in x] for yj in y])
            return X, Y

        @classmethod
        def gradient(cls, Z, *varargs):
            dy = varargs[0] if len(varargs) > 0 else 1.0
            dx = varargs[1] if len(varargs) > 1 else 1.0
            rows = len(Z)
            cols = len(Z[0])
            dZdx = []
            dZdy = []
            for r in range(rows):
                row_dx = []
                for c in range(cols):
                    if c == 0:
                        row_dx.append((Z[r][1] - Z[r][0]) / dx)
                    elif c == cols - 1:
                        row_dx.append((Z[r][-1] - Z[r][-2]) / dx)
                    else:
                        row_dx.append((Z[r][c + 1] - Z[r][c - 1]) / (2.0 * dx))
                dZdx.append(row_dx)
            for r in range(rows):
                row_dy = []
                for c in range(cols):
                    if r == 0:
                        row_dy.append((Z[1][c] - Z[0][c]) / dy)
                    elif r == rows - 1:
                        row_dy.append((Z[-1][c] - Z[-2][c]) / dy)
                    else:
                        row_dy.append((Z[r + 1][c] - Z[r - 1][c]) / (2.0 * dy))
                dZdy.append(row_dy)
            return cls.array(dZdy), cls.array(dZdx)

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


# --- Analytical Benchmark Models ---

def rocket_velocity(t):
    """
    Analytical rocket velocity benchmark:
    v(t) = u * ln(m0 / (m0 - q * t)) - g * t
    where:
      u  = 2000.0 m/s  (exhaust velocity)
      m0 = 150000.0 kg (initial rocket mass)
      q  = 2500.0 kg/s (fuel consumption rate)
      g  = 9.81 m/s^2  (gravitational acceleration)
    """
    u = 2000.0
    m0 = 150000.0
    q = 2500.0
    g = 9.81
    t_arr = np.asarray(t, dtype=float)
    res = u * np.log(m0 / (m0 - q * t_arr)) - g * t_arr
    if np.isscalar(t):
        return float(res)
    return res


def rocket_acceleration(t):
    """
    Exact analytical rocket acceleration (time derivative of rocket_velocity):
    a(t) = (u * q) / (m0 - q * t) - g
    """
    u = 2000.0
    m0 = 150000.0
    q = 2500.0
    g = 9.81
    t_arr = np.asarray(t, dtype=float)
    res = (u * q) / (m0 - q * t_arr) - g
    if np.isscalar(t):
        return float(res)
    return res


# --- Finite Difference Stencils ---

def forward_difference(f, x, h):
    """
    First-order forward finite difference stencil:
    f'(x) ≈ (f(x + h) - f(x)) / h
    """
    if callable(f):
        return (f(x + h) - f(x)) / h
    raise TypeError("f must be a callable function f(x)")


def backward_difference(f, x, h):
    """
    First-order backward finite difference stencil:
    f'(x) ≈ (f(x) - f(x - h)) / h
    """
    if callable(f):
        return (f(x) - f(x - h)) / h
    raise TypeError("f must be a callable function f(x)")


def central_difference(f, x, h):
    """
    Second-order central finite difference stencil:
    f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)
    """
    if callable(f):
        return (f(x + h) - f(x - h)) / (2.0 * h)
    raise TypeError("f must be a callable function f(x)")


def second_derivative(f, x, h):
    """
    Second-order central difference for the second derivative:
    f''(x) ≈ (f(x + h) - 2 * f(x) + f(x - h)) / h^2
    """
    if callable(f):
        return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h ** 2)
    raise TypeError("f must be a callable function f(x)")


# Stencil aliases for flexibility
forward_diff = forward_difference
backward_diff = backward_difference
central_diff = central_difference
second_central_diff = second_derivative


# --- Interactive Plotly Visualizers ---

def plot_secant_tangent_limit(f, x0=2.0, h_values=(1.0, 0.5, 0.2, 0.05), x_range=(0.5, 4.0), title="Secant to Tangent Convergence"):
    """
    Interactive Plotly figure illustrating the geometric limit of secant lines
    approaching the true tangent line as step size h -> 0.
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    y = f(x)
    y0 = f(x0)

    # High-accuracy numerical slope for true tangent reference
    eps = 1e-6
    true_slope = (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps)
    tangent_line = y0 + true_slope * (x - x0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name='f(x)',
        line=dict(color='#111827', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=tangent_line, mode='lines',
        name=f"True Tangent (slope = {true_slope:.4f})",
        line=dict(color='#DC2626', width=2.5, dash='dash')
    ))

    colors = ['#2563EB', '#D97706', '#059669', '#7C3AED', '#DB2777']
    for idx, h in enumerate(h_values):
        x1 = x0 + h
        y1 = f(x1)
        slope = (y1 - y0) / h
        secant_line = y0 + slope * (x - x0)
        c = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=x, y=secant_line, mode='lines',
            name=f"Secant h={h} (slope = {slope:.4f})",
            line=dict(color=c, width=1.8, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=[x1], y=[y1], mode='markers',
            name=f"f(x0 + {h})",
            marker=dict(color=c, size=7, symbol='diamond'),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=[x0], y=[y0], mode='markers',
        name=f"Anchor Point (x0, f(x0))",
        marker=dict(color='#111827', size=11, symbol='circle')
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


def plot_derivative_comparison(t, a_true, approximations, title="Numerical vs Analytical Derivative"):
    """
    Compares numerical derivatives against analytical truth using Plotly subplots.
    t: 1D array of evaluation coordinates (e.g. time)
    a_true: 1D array of exact analytical derivative values
    approximations: dict of {'Scheme Label': numerical_derivative_array}
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Derivative Comparison", "Absolute Error |Approx - True|"),
        row_heights=[0.65, 0.35]
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=a_true, mode='lines', name='Analytical True',
            line=dict(color='#111827', width=2.5)
        ),
        row=1, col=1
    )

    palette = ['#2563EB', '#059669', '#D97706', '#7C3AED', '#DC2626', '#0891B2']
    t_list = list(t)
    a_true_list = [float(val) for val in a_true]

    for idx, (label, a_num) in enumerate(approximations.items()):
        c = palette[idx % len(palette)]
        a_num_list = [float(val) for val in a_num]
        fig.add_trace(
            go.Scatter(
                x=t, y=a_num, mode='lines+markers', name=label,
                line=dict(color=c, width=1.5), marker=dict(size=4)
            ),
            row=1, col=1
        )
        err_clean = [max(abs(an - at), 1e-16) for an, at in zip(a_num_list, a_true_list)]
        fig.add_trace(
            go.Scatter(
                x=t, y=err_clean, mode='lines', name=f"{label} Error",
                line=dict(color=c, width=1.5)
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Derivative f'(t)", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Error", type="log", row=2, col=1)
    fig.update_xaxes(title_text="t", row=2, col=1)
    return fig


def plot_error_scaling(h_values, errors_dict, expected_orders=None, title="Finite Difference Error vs Step Size h"):
    """
    Log-log Plotly convergence plot showing error decay against step size h.
    h_values: 1D array-like of step sizes
    errors_dict: dict of {'Scheme Name': errors_array}
    expected_orders: optional dict of {'Scheme Name': int_order}
    """
    h_arr = np.asarray(h_values, dtype=float)
    fig = go.Figure()
    palette = ['#2563EB', '#059669', '#D97706', '#7C3AED', '#DC2626']

    for idx, (name, errs) in enumerate(errors_dict.items()):
        c = palette[idx % len(palette)]
        err_arr = np.asarray(errs, dtype=float)
        fig.add_trace(go.Scatter(
            x=h_arr, y=err_arr, mode='lines+markers', name=name,
            line=dict(color=c, width=2), marker=dict(size=7, symbol='circle')
        ))

    if expected_orders:
        for name, order in expected_orders.items():
            if name in errors_dict:
                y0 = errors_dict[name][0]
                h0 = h_arr[0]
                ref_line = (h_arr ** order) * (y0 / (h0 ** order))
                fig.add_trace(go.Scatter(
                    x=h_arr, y=ref_line, mode='lines',
                    name=f"Theoretical O(h^{order})",
                    line=dict(dash='dash', color='#6B7280', width=1.3)
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Step Size h",
        yaxis_title="Absolute Error",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    return fig


def plot_surface_gradient_3d(func, x_range=(-2, 2), y_range=(-2, 2), n_grid=40, arrow_stride=4, title="3D Surface and Numerical Gradient Vector Field", subplots=False):
    """
    Interactive 3D Plotly visualization of a 2D scalar field z = f(x, y)
    with 3D gradient vector cones, optionally expanded to subplots for
    gradient components in x and y.
    """
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dZdx = np.gradient(Z, dx, axis=1)
    dZdy = np.gradient(Z, dy, axis=0)

    # Downsampled cone vectors
    Xs = X[::arrow_stride, ::arrow_stride].flatten()
    Ys = Y[::arrow_stride, ::arrow_stride].flatten()
    Zs = Z[::arrow_stride, ::arrow_stride].flatten()
    Us = dZdx[::arrow_stride, ::arrow_stride].flatten()
    Vs = dZdy[::arrow_stride, ::arrow_stride].flatten()
    Ws = np.asarray([0.0 for _ in Us])

    if subplots:
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
            subplot_titles=(
                "Surface f(x, y) with ∇f Cones",
                "Gradient Component: ∂f/∂x",
                "Gradient Component: ∂f/∂y"
            ),
            horizontal_spacing=0.03
        )
        # Panel 1: Surface with gradient cones
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.85,
            name='f(x, y)', showscale=False
        ), row=1, col=1)
        fig.add_trace(go.Cone(
            x=Xs, y=Ys, z=Zs, u=Us, v=Vs, w=Ws,
            sizemode='scaled', sizeref=0.35,
            colorscale='Blues', showscale=False,
            name='∇f'
        ), row=1, col=1)

        # Panel 2: x component ∂f/∂x
        fig.add_trace(go.Surface(
            x=X, y=Y, z=dZdx, colorscale='RdBu', showscale=False,
            name='∂f/∂x'
        ), row=1, col=2)

        # Panel 3: y component ∂f/∂y
        fig.add_trace(go.Surface(
            x=X, y=Y, z=dZdy, colorscale='RdBu', showscale=False,
            name='∂f/∂y'
        ), row=1, col=3)

        fig.update_layout(
            title=title,
            height=550,
            template="plotly_white",
            margin=dict(l=10, r=10, b=10, t=50)
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.85,
        name='f(x, y)', showscale=False
    ))

    fig.add_trace(go.Cone(
        x=Xs, y=Ys, z=Zs, u=Us, v=Vs, w=Ws,
        sizemode='scaled', sizeref=0.35,
        colorscale='Reds', showscale=False,
        name='Gradient ∇f'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)"
        ),
        template="plotly_white",
        margin=dict(l=10, r=10, b=10, t=40)
    )
    return fig


def plot_noise_amplification(t, clean_signal, noisy_signal, d_clean, d_noisy, title="Noise Amplification in Numerical Differentiation"):
    """
    Two-panel Plotly display contrasting clean vs noisy raw signals and their derivatives.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("Original vs Noisy Signal", "Analytical vs Finite Difference Derivative")
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=clean_signal, mode='lines', name='Clean Signal',
            line=dict(color='#2563EB', width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=noisy_signal, mode='lines', name='Noisy Signal',
            line=dict(color='#9CA3AF', width=1, dash='dot')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=d_clean, mode='lines', name='Clean Derivative',
            line=dict(color='#2563EB', width=2)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=d_noisy, mode='lines', name='Noisy Derivative',
            line=dict(color='#DC2626', width=1)
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Signal f(t)", row=1, col=1)
    fig.update_yaxes(title_text="Derivative f'(t)", row=2, col=1)
    fig.update_xaxes(title_text="Time t", row=2, col=1)
    return fig


def plot_frequency_noise_spectrum(frequencies, fft_signal, fft_derivative, title="Frequency Spectrum of Signal vs Derivative Noise"):
    """
    Plotly frequency spectrum safely filtering positive frequencies to eliminate zero-division warnings.
    """
    freqs = list(frequencies)
    pos_indices = [i for i, f in enumerate(freqs) if f > 0]
    f_pos = [freqs[i] for i in pos_indices]
    sig_mag = [math.fabs(float(fft_signal[i])) for i in pos_indices]
    deriv_mag = [math.fabs(float(fft_derivative[i])) for i in pos_indices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_pos, y=sig_mag, mode='lines', name='Signal Magnitude |X(f)|',
        line=dict(color='#2563EB', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=f_pos, y=deriv_mag, mode='lines', name='Derivative Magnitude |(2πif)X(f)|',
        line=dict(color='#DC2626', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency f (Hz)",
        yaxis_title="Magnitude",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    return fig


def plot_uneven_spacing_derivative(x, y, dy_num, dy_true=None, title="Derivative on Unevenly Spaced Grid"):
    """
    Plotly comparison showing derivative evaluation on an irregularly sampled grid.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Uneven Grid Samples f(x)", "Numerical Derivative f'(x)")
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode='lines+markers', name='f(x) Samples',
            line=dict(color='#2563EB', width=1.5),
            marker=dict(size=6, symbol='circle')
        ),
        row=1, col=1
    )

    if dy_true is not None:
        fig.add_trace(
            go.Scatter(
                x=x, y=dy_true, mode='lines', name="True f'(x)",
                line=dict(color='#111827', width=2)
            ),
            row=2, col=1
        )
    fig.add_trace(
        go.Scatter(
            x=x, y=dy_num, mode='lines+markers', name="Uneven Stencil f'(x)",
            line=dict(color='#DC2626', width=1.5),
            marker=dict(size=6, symbol='cross')
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_yaxes(title_text="f'(x)", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    return fig
