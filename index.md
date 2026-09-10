# Welcome to Executable Engineering - Interactive Numerical Methods in Python

```{image} images/exe_logo.png
:width: 60%
:align: center
```

The core philosphy of this text it to experience first-hand the use of numerical methods to solve mathematical problems. Herein, we'll explore how complex problems are decomposed into simpler, easier problems that can be solved algorithmically. We will discuss the fundametal principles used in modern open-source tools and how these tools can be leveraged efficiently and effectively.


**NB:** This book is designed to be interactive in-browser. However, this technology is evolving and not always stable. While execution of the cells appears robust, editing is temperamental. As a fallback, a Google Colab link is provided for each notebook which will launch a fully interactive Python environment.

## Tools and Resources

* [PDF of the full book](/ExecutableEngineering/book.pdf)
* [Youtube channel with lectures](https://www.youtube.com/@McMaster_EngPhys3NM4)
* [Virtual Tutor](https://notebooklm.google.com/notebook/31dca965-3ce5-42e2-b423-c04bd11bfc98)
* [Associated PyPI Package](https://pypi.org/project/executable-engineering/)

This resource is being developed in part through the [McMaster University Open Educational Resources grant](https://mi.mcmaster.ca/oer-grant).

## Course Preamble & Timetable
The course is paced for appximately a 12 week program, with 3 lectures per week. 

* **[Week 1: Truncation and Round-off Error](chapters/numerical_error/numerical_error.ipynb)**
  * **Day 1:** Course Introduction
  * **Day 2:** Round-off error and finite precision representations
  * **Day 3:** True and approximate error, and truncation error
* **[Week 2: Linear Systems](chapters/linear_systems/linear_systems.ipynb)**
  * **Day 4:** Solvability (Over/Under-determined systems & the Pseudo-Inverse), Conditioning, and Complexity
  * **Day 5:** Direct methods: Gaussian elimination, Diagonal dominance, and pivoting
  * **Day 6:** Direct methods: LU decomposition, Matrix types, and sparsity
* **[Week 3: Linear Systems (Continued)](chapters/linear_systems/linear_systems.ipynb)**
  * **Day 7:** Iterative methods: Stationary methods
  * **Day 8:** Iterative methods: Krylov subspace methods
  * **Day 9:** Iterative methods: Preconditioners
* **[Week 4: Interpolation and Curve Fitting](chapters/interpolation_and_curve_fitting/interpolation_and_curve_fitting.ipynb)**
  * **Day 10:** Polynomial interpolation
  * **Day 11:** Splines, Radial basis functions, and Summary of interpolation
  * **Day 12:** Curve fitting (Linear least squares, Weighted, Polyfit, Best fit RBF)
* **[Week 5: Roots of Equations](chapters/root_finding/root_finding.ipynb)**
  * **Day 13:** Polynomial roots and Closed methods
  * **Day 14:** Open methods (Newton-Raphson)
  * **Day 15:** Basins of attraction and Global convergence methods
* **[Week 6: Optimization](chapters/optimization/optimization.ipynb)**
  * **Day 16:** Closed methods and Open methods (Gradient-based optimizers)
  * **Day 17:** Indefinite systems and Trust region methods
  * **Day 18:** Equality constraints, Nonlinear least squares, and Neural networks
* **Midterm Break**
* **[Week 7: Numerical Differentiation](chapters/differentiation_and_integration/differentiation_and_integration.md)**
  * **Day 19:** Finite difference (First-order derivatives)
  * **Day 20:** Finite difference (Second-order and higher dimensions)
  * **Day 21:** Even vs. uneven spacing, and Uncertainty amplification
* **[Week 8: Numerical Integration](chapters/differentiation_and_integration/integration/numerical_integration.ipynb)**
  * **Day 22:** Newton-Cotes formulae (Riemann's integrals, Trapezoid rule, Simpson's rules)
  * **Day 23:** Romberg rule
  * **Day 24:** Gaussian quadrature
* **[Week 9: Differential Equations (Initial Value Problems)](chapters/differential_equations/initial_value_problems/initial_value_problems.ipynb)**
  * **Day 25:** Explicit methods (Runge-Kutta) and Adaptive time stepping
  * **Day 26:** Implicit methods, Stiffness, and Implicit Runge-Kutta
  * **Day 27:** Multistep methods, Systems of equations, and Reduction of order
* **[Week 10: Boundary Value Problems and PDEs](chapters/differential_equations/boundary_value_problems/boundary_value_problems.ipynb)**
  * **Day 28:** Shooting method and Collocation methods
  * **Day 29:** Finite difference method and Finite volume
  * **Day 30:** Partial differential equations
* **[Week 11: Finite Element Method](chapters/finite_element_method/finite_element_method.ipynb)**
  * **Day 31:** Motivating example and Theoretical background
  * **Day 32:** Elements, Discretization, and Assembly
  * **Day 33:** Solution methods
* **[Week 12: Eigenvalue Problems](chapters/eigenvalue_problems/eigenvalue_problems.ipynb)**
  * **Day 34:** Eigenfunctions and Powers method
  * **Day 35:** Matrix decomposition methods
  * **Day 36:** Package tools
* **Week 13: Project Presentations & Review**
  * **Day 37:** Project Presentations I
  * **Day 38:** Project Presentations II
  * **Day 39:** Final Course Review

## A bit of fun

![Meme](images/Meme.png)
