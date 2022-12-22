import numpy as np
from sympy import lambdify
import sympy as sp

class NewtonSolver():
    def __init__(self, initialGuess, f, JacobianMethod, residualSmoothing, variables, **kwargs) -> None:
        """
        initialGuess = [x1, x2, x3, ..., xN]
        f = [func1, func2, func3, ..., funcN]
        JacobianMethod = "Analytical" or "FD"
        """
        tol = 5e-8
        
        nVars = len(initialGuess)
        guess = np.array(initialGuess)
        guess = guess.reshape((1, nVars))

        ### Find out how far away from root we are
        f_vec = np.zeros((1, nVars)) #Row vector
        #print("initial f_vec", f_vec)
        
        
        ### "'symbolise' all functions once instead of every step"
        for i in range(nVars):
            f[i] = f[i](*variables, kwargs)
        for i in range(nVars):
            f_callable = lambdify(variables, f[i])
            f_vec[0][i] = f_callable(*guess[0])
        
        #print("filled f_vec", f_vec)
        f_len = np.linalg.norm(f_vec)
        print("Error: ", f_len)
        #print("guess after reshape", guess)
        loop = 1
        #print("entering while loop")
        alpha = 1.0
        while f_len > tol:
            print(loop)
            if nVars == 1: #1D problem so don't need Jacobian
                df_dx = lambdify(variables, f[0].diff(variables[0]))
                df_dx_val = df_dx(*guess[0])
                guess -= np.divide(f_vec, df_dx_val)
                pass
            else: #Multi-dimension problem, so need Jacobian matrix
                Jac = self.FormJacobian(f = f, JacobianMethod = JacobianMethod, kwargs = kwargs, variables = variables, guess = guess)
                #print("Jacobian", Jac)
                J_inv = np.linalg.inv(Jac)
                #print(np.shape(J_inv), np.shape(np.transpose(f_vec)))
                #print(np.shape(np.transpose(f_vec)))
                #print("guess shape", np.shape(guess))
                #print("J_inv shape", np.shape(J_inv))
                #print("transpose f_vec shape", np.shape(np.transpose(f_vec)))
                #print("J_inv * f_vec^T shape", np.shape(np.matmul(J_inv, np.transpose(f_vec))))
                #print("update shape", np.shape(np.transpose(np.multiply(J_inv, np.transpose(f_vec)))))
                if residualSmoothing[0]:
                    alpha = residualSmoothing[1]
                guess -= alpha * np.transpose(np.matmul(J_inv, np.transpose(f_vec)))
                #print("new guess", guess)
                
            for i in range(nVars):
                f_callable = lambdify(variables, f[i])
                f_vec[0][i] = f_callable(*guess[0])
            f_len = np.linalg.norm(f_vec)
            print("Error: ", f_len)
            #print("new f_vec", f_vec)
            loop += 1
        self.final_result = guess[0].tolist()
        
        

    def FormJacobian(self, f, JacobianMethod, variables, kwargs, guess):
        epsilon = 1e-9
        nVars = len(variables)
        Jac = np.zeros((nVars, nVars))
        for i in range(nVars): #counter to switch between functions
            for j in range(nVars): #counter to switch between variables
                if JacobianMethod == "Analytical":
                    df_dxi = lambdify(variables, f[i].diff(variables[j]))
                    df_dxi_val = df_dxi(*guess[0])
                    
                elif JacobianMethod == "FD":
                    pass
                Jac[i][j] = df_dxi_val

        return Jac