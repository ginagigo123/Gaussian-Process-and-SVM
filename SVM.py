# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:03:29 2021

@author: ginag
"""

import numpy as np
from libsvm.svmutil import *
import time # calculate the compute time

def openCsv(filePath):
    data = np.genfromtxt(filePath, delimiter=',')
    return data

def findBestParameter(Y_train, X_train, optimal_acc, optimal_command, command):
    now_acc = svm_train(Y_train, X_train, command)
    if optimal_acc < now_acc:
        return now_acc, command
    else:
        return optimal_acc, optimal_command
    
def gridSearch(X_train, Y_train):
    cost = [1e-3, 1e-2, 1e-1, 1, 10]
    gamma = [ 1e-5, 1/784, 1e-2, 1]
    degree = [1, 2, 3]
    coef = [0, 1, 2, 3]
    opt_cmd = ''
    opt_acc = 0
    
    for k in kernel:
        for c in cost:
            if k == "linear":
                command = f"-t {kernel[k]} -c {c} -v 10"
                opt_acc, opt_cmd = findBestParameter(Y_train, X_train, opt_acc, opt_cmd, command)
                print(k," opt_acc : ", opt_acc)
                print() 
                
            if k == "polynomial":
                for g in gamma:
                    for d in degree:
                        for co in coef:
                          print(k, "gamma:", g, "d", d, "coef0", co)
                          command = f"-t {kernel[k]} -c {c} -g {g} -d {d} -v 10 -r {co}"
                          opt_acc, opt_cmd = findBestParameter(Y_train, X_train, opt_acc, opt_cmd, command)
                          print(k," opt_acc : ", opt_acc)
                          print()
            if k == "RBF":
                for g in gamma:
                    print(k, "cost:", c, "gamma:", g)
                    command = f"-t {kernel[k]} -c {c} -g {g} -v 10"
                    opt_acc, opt_cmd = findBestParameter(Y_train, X_train, opt_acc, opt_cmd, command)
                    print(k," opt_acc : ", opt_acc)
                    print()    
    return opt_acc, opt_cmd
    
def linearKernel(u, v):
    return u @ v.T

def RBFKernel(u, v, gamma):
    dist = np.sum(u ** 2, axis=1).reshape(-1, 1) + np.sum(v **2, axis=1) - 2 * u @ v.T
    return np.exp(-gamma *  dist)

kernel = {
    'linear': 0,
    'polynomial': 1,
    'RBF': 2
    }

X_train = openCsv('./data/X_train.csv')
Y_train = openCsv('./data/Y_train.csv')

X_test = openCsv('./data/X_test.csv')
Y_test = openCsv('./data/Y_test.csv')

# Part one
for k in kernel:
    start = time.time()
    command = f"-t {kernel[k]}"
    m = svm_train(Y_train, X_train, command)
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
    end = time.time()
    print(k, " total time:", end - start)
    print()

# Part two
'''
 use C-SVC:
     do grid search for finding the parameter
'''
opt_acc, opt_cmd= gridSearch(X_train, Y_train)

opt_cmd = opt_cmd[:-5]
opt_model = svm_train(Y_train, X_train, opt_cmd)
opt_label, opt_test_acc, opt_val = svm_predict(Y_test, X_test, opt_model) 


# Part three
'''
linear kernel + rbf kernel together => a new kernel function

'''
combineKernel = linearKernel(X_train, X_train) + RBFKernel(X_train, X_train, 1/784)
combineKernel_test = linearKernel(X_test, X_train) + RBFKernel(X_test, X_train, 1/784)

X_kernel = np.hstack((np.arange(1, 5001).reshape(-1, 1), combineKernel))
X_kernel_s = np.hstack((np.arange(1, 2501).reshape(-1, 1), combineKernel_test))

model = svm_train(Y_train, X_kernel, "-t 4")
p_label, p_acc, p_val = svm_predict(Y_test, X_kernel_s, model)
