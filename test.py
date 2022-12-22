from sympy.abc import x, y
import sympy as sp
from sympy import lambdify
import numpy as np

def test(a, b, **kwargs):
    print(a + b)

z = [0.0, 1.0]
#z = np.array(z)
def test2(x, y, **kwargs):
    k = kwargs["k"]
    alpha = kwargs["alpha"]
    return k * x ** 2 * y + alpha * (y - x)

f = test2(x, y, k = 2.0, alpha = 5.0)
df_dx = lambdify((x, y), f.diff(x))
df_dy = lambdify((x, y), f.diff(y))

print(df_dx(*z))
print(df_dy(*z))
