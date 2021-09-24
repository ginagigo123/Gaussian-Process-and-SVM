
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:00:24 2021

@author: ginagigo
"""

import numpy as np
from numpy.linalg import inv, cholesky, det
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

def read_input():
    # read file
    data = np.zeros(shape=(34, 2))
    index = 0
    with open(".\data\input.data", "r") as f:
        for line in f.readlines():
            data[index, 0] = line.split(" ")[0]
            data[index, 1] = line.split(" ")[1]
            index += 1
    f.close()
    X = np.array(data[:, 0]).reshape(-1, 1)
    Y = np.array(data[:, 1]).reshape(-1, 1)
    return X, Y

def rational_quadratic_kernel_method(Xa, Xb, var, alpha, length_scale):
    sqdist = np.sum(Xa ** 2, axis=1).reshape(-1, 1) + np.sum(Xb **2, axis=1) - 2 * Xa @ Xb.T
    kernel = var * ( 1 + sqdist / (2 * alpha * (length_scale ** 2) )) ** (- alpha)
    return kernel

def calculate_Covar(n, X, beta, var, alpha, length_scale):
    C = np.zeros(shape=(n, n), dtype=float)
    kernel = rational_quadratic_kernel_method(X, X, var, alpha, length_scale)
    C = kernel + (1 / beta) * np.eye(n)
    return C

def Gaussian_Process(X, Y, X_pos, beta, var, alpha, length_scale):
    # compute the Covariance: (34, 34) 
    C = calculate_Covar(X.shape[0], X, beta, var, alpha, length_scale)

    # draw a line to represent the mean of f in range[-60, 60]
    kernel_upR = rational_quadratic_kernel_method(X, X_pos, var, alpha, length_scale)
    kernel_downR = rational_quadratic_kernel_method(X_pos, X_pos, var, alpha, length_scale)
    kernel_downR += ( 1 / beta ) * np.eye(X_pos.shape[0])
    
    mean_point = kernel_upR.T @ inv(C) @ Y
    var_point = kernel_downR
    var_point -= kernel_upR.T @ inv(C) @ kernel_upR
    return mean_point, var_point
 
def plot_figure(X, Y, X_pos, mean_pos, var_pos, args, text=""):
    standard = np.diag(var_pos) ** (1/2)
    standard = standard.reshape(-1, 1)
    plt.plot(X_pos, mean_pos, "b")
    plt.scatter(X, Y, c="#ff00f2")
    
    plt.plot(X_pos, mean_pos + 1.96 * standard, "r")
    plt.plot(X_pos, mean_pos - 1.96 * standard, "r")
    
    upper = mean_pos + 1.96 * standard
    lower = mean_pos - 1.96 * standard
    plt.fill_between(X_pos[:, 0], upper[:, 0], lower[:, 0], alpha = 0.2, color="#ffe294")
    plt.title(text + " beta: {:.2f}, var: {:.2f}, alpha: {:.2f}, length_scale: {:.2f}".format(args[0], args[1], args[2], args[3]))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(text+".png")

   
def costFunction_naive(theta, X, Y, beta):
    """
    Returns the function that compute the negative log marginal 
    likelihood for training X and y
    """
    theta = theta.ravel()
    var, alpha, length_scale = theta[0], theta[1], theta[2]
    
    C_opt = calculate_Covar(X.shape[0], X, beta, var, alpha, length_scale)
    
    first =  1 / 2 * np.log(det(C_opt))
    second =  1/ 2 * Y.T @ inv(C_opt) @ Y
    third =  X.shape[0] / 2 * np.log(2 * np.pi)
    result = first + second + third
    return result.ravel() 

def costFunction_stable(theta, X, Y, beta):
    """
    Returns the function that compute the negative log marginal 
    likelihood for training X and y
    """
    theta = theta.ravel()
    var, alpha, length_scale = theta[0], theta[1], theta[2]
    
    C_opt = calculate_Covar(X.shape[0], X, beta, var, alpha, length_scale)
    L = cholesky(C_opt)
    
    S1 = solve_triangular(L, Y, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)
        
    first = np.sum(np.log(np.diagonal(L)))
    second = 1/ 2 * Y.T @ S2
    third = X.shape[0] / 2 * np.log(2 * np.pi)
    result = first + second + third
    return result.ravel() 
    
# read the input.data  X: (34,1 ) Y: (34, 1) 
X, Y = read_input()

# Apply Gaussian Process Regression to predict the distribution of f
beta = 5
var = 10
alpha = 1
length_scale = 1

X_point = np.linspace(-60, 60, 1000).reshape(-1, 1)

# Gaussian Process with initial value 
mean_1, var_1 = Gaussian_Process(X, Y, X_point, beta, var, alpha, length_scale)           
plot_figure(X, Y, X_point, mean_1, var_1, [beta, var, alpha, length_scale], "Test 1")
    
# optimize the hyperparameter
result = minimize(costFunction_naive, [var, alpha, length_scale], args=(X, Y, beta), bounds=((1e-8, 1e6), (1e-4, 1e6), (1e-8, 1e6)) )
var_opt = result.x[0]
alpha_opt = result.x[1]
length_scale_opt = result.x[2]

# Gaussian Process with initial value 
mean_opt, point_var_opt = Gaussian_Process(X, Y, X_point, beta, var_opt, alpha_opt, length_scale_opt)
plot_figure(X, Y, X_point, mean_opt, point_var_opt, 
            [beta, var_opt, alpha_opt, length_scale_opt], "Optimal")


# stable way to optimize the hyperparameter
result = minimize(costFunction_stable, [var, alpha, length_scale], args=(X, Y, beta), bounds=((1e-8, 1e6), (1e-4, 1e6), (1e-8, 1e6)) )
var_opt = result.x[0]
alpha_opt = result.x[1]
length_scale_opt = result.x[2]

mean_opt, point_var_opt = Gaussian_Process(X, Y, X_point, beta, var_opt, alpha_opt, length_scale_opt)
plot_figure(X, Y, X_point, mean_opt, point_var_opt, 
            [beta, var_opt, alpha_opt, length_scale_opt], "Stable Opt")