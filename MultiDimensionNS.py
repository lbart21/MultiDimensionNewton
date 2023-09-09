import numpy as np
from sympy import lambdify
import sympy as sp
from copy import deepcopy

class NewtonSolver():
    def __init__(self, initial_guess, f, jacobian_method, residual_smoothing, \
                    variables, verbose, **kwargs) -> None:
        """

        Parameters
        ----------
        initial_guess : list(float)
            initial guess of the desired variables, in the order of the variables given in the "variables" argument
        f : list(function(s))
            list of function names that are to be solved.
        jacobian_method : string("Analytical" or "FD")
            String for case switching between analytical and finite difference evaluation of the jacobian.
        residual_smoothing : list(boolean, float)
            Toggle step size with the first element, then supply a float as the second element if toggle is true.
        variables: list(sympy character(s))
            list of sympy symbolic characters that are to be solved for
        verbose : boolean
            Toggle current error value being printed after each iteration.
        kwargs : floats
            Additional parameters that are used when evaluating function calls of the functions within f.
        
        Outputs
        ---------
        Converged result is stored in object property "final_result", which is a list.
        """
        tol = 1e-8
        
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
                    guess_plus_epsilon = deepcopy(guess)
                    guess_plus_epsilon[0][j] += epsilon
                    #print(guess, guess_plus_epsilon)
                    f_callable = lambdify(variables, f[i])
                    df_dxi_val = (f_callable(*guess_plus_epsilon[0]) - f_callable(*guess[0])) / epsilon
                jac[i][j] = df_dxi_val

        return jac