from Algorithms.MultiDimensionNewton.MultiDimensionNS import NewtonSolver
from sympy.abc import x, y

def f1(x, y, kwargs):
    k = kwargs["k"]
    a = kwargs["a"]
    h = kwargs["h"]
    term1 = k * x ** 2
    term2 = a * y ** 2
    return term1 + term2 - h

def f2(x, y, kwargs):
    c = kwargs["c"]
    b = kwargs["b"]
    term1 = c * x
    term2 = b * y
    return term1 - term2

def f3(x, kwargs):
    a = kwargs["a"]
    b = kwargs["b"]

    return (x - a) * (x - b)

A = NewtonSolver(initial_guess = [1.0, 2.0], f = [f1, f2], jacobian_method = "Analytical", variables = [x, y], k = 1.0, a = 1.0, b = 1.0, c = 1.0, h = 0.0, residual_smoothing = [False], verbose = [True])
#A = NewtonSolver(initialGuess = [3.0], f = [f3], JacobianMethod = "Analytical", residualSmoothing = [False], variables = [x], a = 2.0, b = 5.0)
result1 = A.final_result
print(*result1)
print("Starting FD test")
B = NewtonSolver(initial_guess = [1.0, 2.0], f = [f1, f2], jacobian_method = "FD", variables = [x, y], k = 1.0, a = 1.0, b = 1.0, c = 1.0, h = 0.0, residual_smoothing = [False], verbose = [False])
result2 = B.final_result
print(*result2)