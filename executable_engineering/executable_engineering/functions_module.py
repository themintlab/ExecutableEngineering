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
