try:
    from .neuralnet_module import init_weights, tanh, tanh_derivative, forward, compute_loss, backward, plot_nn_diagram, init_model, forward_pass, backward_pass, step, reset_model, save_function, change_depth, change_width, draw_network, update_plots, interactive_neural_network, train_until_convergence
except ImportError:
    pass

try:
    from .openrootfinder_module import RootFinderOpen
except ImportError:
    pass

try:
    from .closedrootfinder_module import RootFinderClosed
except ImportError:
    pass

try:
    from .closedoptimization_module import OptimizerClosed, ScalarTracker, plot_bracketed_progression
except ImportError:
    pass

try:
    from .openoptimization_module import OptimizerOpen, plot_optimization_path_2d, plot_optimization_path_3d
except ImportError:
    pass

try:
    from .gradientoptimization_module import OptimizerGrad
except ImportError:
    pass

try:
    from .numerical_error_module import python_internal_binary, decimal_to_binary
except ImportError:
    pass

try:
    from .linearsystems_module import visual_solve_2d, visualize_conditioning, visualize_matrix_norms, jacobi_step, gauss_seidel_step, sor_step, visualize_convergence_2d, create_diagonally_dominant_matrix, iter_solve, steepest_descent, conjugate_gradient, IterationTracker, discretise_poisson, spai
except ImportError:
    pass

try:
    from .functions_module import weierstrass, plot_and_find_roots, complex_plot_with_shadow, plot_polynomial_roots_companion, plot_secant, plot_newton_raphson, plot_basins_of_attraction, newton_raphson
except ImportError:
    pass

try:
    from .optimization_module import visual_golden_section, visual_powell_2d, newton_method
except ImportError:
    pass

try:
    from .constrainedoptimization_module import plot_constrained_optimum_3d
except ImportError:
    pass
from .differentiation_module import (
    rocket_velocity,
    rocket_acceleration,
    forward_difference,
    backward_difference,
    central_difference,
    second_derivative,
    forward_diff,
    backward_diff,
    central_diff,
    second_central_diff,
    plot_secant_tangent_limit,
    plot_derivative_comparison,
    plot_error_scaling,
    plot_surface_gradient_3d,
    plot_noise_amplification,
    plot_frequency_noise_spectrum,
    plot_uneven_spacing_derivative,
)
from .integration_module import (
    trapezoid_rule,
    simpson_13_rule,
    simpson_38_rule,
    romberg_integration,
    gauss_legendre_quadrature,
    composite_trapezoid,
    simpson13,
    simpson38,
    simpson_rule,
    plot_area_under_curve,
    plot_riemann_sum,
    plot_trapezoid_rule,
    plot_simpson_rule,
    plot_romberg_convergence,
    plot_gaussian_quadrature,
)

__all__ = [
  "interactive_neural_network",
  "train_until_convergence",
  "plot_constrained_optimum_3d",
  "plot_optimization_path_3d",
  "plot_optimization_path_2d",
  "ScalarTracker",
  "plot_bracketed_progression",
  "newton_method",
  "visual_powell_2d",
  "visual_golden_section",
  "plot_polynomial_roots_companion",
  "plot_secant",
  "plot_newton_raphson",
  "plot_basins_of_attraction",
  "newton_raphson",
  "plot_and_find_roots",
  "complex_plot_with_shadow",
  "weierstrass",
  "discretise_poisson",
  "spai",
  "visual_solve_2d",
  "visualize_conditioning",
  "visualize_matrix_norms",
  "jacobi_step",
  "gauss_seidel_step",
  "sor_step",
  "visualize_convergence_2d",
  "create_diagonally_dominant_matrix",
  "iter_solve",
  "steepest_descent",
  "conjugate_gradient",
  "IterationTracker",
  "init_weights",
  "tanh",
  "tanh_derivative",
  "forward",
  "compute_loss",
  "backward",
  "plot_nn_diagram",
  "init_model",
  "forward_pass",
  "backward_pass",
  "step",
  "reset_model",
  "save_function",
  "change_depth",
  "change_width",
  "draw_network",
  "update_plots",
  "RootFinderOpen",
  "RootFinderClosed",
  "OptimizerClosed",
  "OptimizerOpen",
  "OptimizerGrad",
  "python_internal_binary",
  "decimal_to_binary",
  # Differentiation module
  "rocket_velocity",
  "rocket_acceleration",
  "forward_difference",
  "backward_difference",
  "central_difference",
  "second_derivative",
  "forward_diff",
  "backward_diff",
  "central_diff",
  "second_central_diff",
  "plot_secant_tangent_limit",
  "plot_derivative_comparison",
  "plot_error_scaling",
  "plot_surface_gradient_3d",
  "plot_noise_amplification",
  "plot_frequency_noise_spectrum",
  "plot_uneven_spacing_derivative",
  # Integration module
  "trapezoid_rule",
  "simpson_13_rule",
  "simpson_38_rule",
  "romberg_integration",
  "gauss_legendre_quadrature",
  "composite_trapezoid",
  "simpson13",
  "simpson38",
  "simpson_rule",
  "plot_area_under_curve",
  "plot_riemann_sum",
  "plot_trapezoid_rule",
  "plot_simpson_rule",
  "plot_romberg_convergence",
  "plot_gaussian_quadrature",
]
