import numpy as np
from sympy import lambdify
import sympy as sp

class NewtonSolver():
    def __init__(self, initial_guess, f, jacobian_method, residual_smoothing, \
                    variables, verbose, **kwargs) -> None:
        """
        initialGuess = [x1, x2, x3, ..., xN]
        f = [func1, func2, func3, ..., funcN]
        JacobianMethod = "Analytical" or "FD"
        """
        tol = 5e-8
        
        n_vars = len(initial_guess)
        guess = np.array(initial_guess)
        guess = guess.reshape((1, n_vars))

        ### Find out how far away from root we are
        f_vec = np.zeros((1, n_vars)) #Row vector

        ### "'symbolise' all functions once instead of every step"
        for i in range(n_vars):
            f[i] = f[i](*variables, kwargs)
        for i in range(n_vars):
            f_callable = lambdify(variables, f[i])
            f_vec[0][i] = f_callable(*guess[0])

        f_len = np.linalg.norm(f_vec)
        if verbose:
            print("Error: ", f_len)

        loop = 1
        alpha = 1.0
        while f_len > tol:
            if verbose:
                print(loop)
            if n_vars == 1: #1D problem so don't need Jacobian
                df_dx = lambdify(variables, f[0].diff(variables[0]))
                df_dx_val = df_dx(*guess[0])
                guess -= np.divide(f_vec, df_dx_val)

            else: #Multi-dimension problem, so need Jacobian matrix
                jac = self.form_jacobian(f = f, jacobian_method = jacobian_method, \
                                            kwargs = kwargs, variables = variables, guess = guess)
                jac_inv = np.linalg.inv(jac)
                if residual_smoothing[0]:
                    alpha = residual_smoothing[1]
                guess -= alpha * np.transpose(np.matmul(jac_inv, np.transpose(f_vec)))

            for i in range(n_vars):
                f_callable = lambdify(variables, f[i])
                f_vec[0][i] = f_callable(*guess[0])
            f_len = np.linalg.norm(f_vec)
            if verbose:
                print("Error: ", f_len)
            loop += 1
        self.final_result = guess[0].tolist()
        
    def form_jacobian(self, f, jacobian_method, variables, kwargs, guess):
        epsilon = 1e-9
        n_vars = len(variables)
        jac = np.zeros((n_vars, n_vars))
        for i in range(n_vars): #counter to switch between functions
            for j in range(n_vars): #counter to switch between variables
                if jacobian_method == "Analytical":
                    df_dxi = lambdify(variables, f[i].diff(variables[j]))
                    df_dxi_val = df_dxi(*guess[0])
                    
                elif jacobian_method == "FD":
                    pass
                jac[i][j] = df_dxi_val

        return jac