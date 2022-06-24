# Machine Learning HW5
## Gaussian Process
### Part a. code with detailed explainations
#### Load data:
```python
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
```
The input data:
```
-5.018038795695388643e+01 1.774746809131112046e+00
-4.833784599279169925e+01 1.130536934559104534e+00
...
```
At first, I load the data by the above code. Split the line, and get X and Y. Reshape the X and Y to be (n, 1). And the n would be 34.

In this experiment we use the ***Rational Quadratic kernel*** function to compute the similarities between different points.
#### Rational quadratic kernel
$\begin{gather}k(x_a, x_b) = \sigma^2(1+\dfrac{||x_a-x_b||^2}{2\alpha l^2})^{-\alpha}\end{gather}$
* $\alpha^2$ the overall variance
* $l$ the lengthscale
* $\alpha$ the scale-mixture ($\alpha$>0)

```python
def rational_quadratic_kernel_method(Xa, Xb, var, alpha, length_scale):
    sqdist = np.sum(Xa ** 2, axis=1).reshape(-1, 1) + np.sum(Xb **2, axis=1) - 2 * Xa @ Xb.T
    kernel = var * ( 1 + sqdist / (2 * alpha * (length_scale ** 2) )) ** (- alpha)
    return kernel
```
Xa and Xb are the numpy array of data points.
The sqdist after calculation would be like $X_a^2 + X_b^2 - 2 X_aX_b$.

As for the var, alpha and length_scale are the parameter of rational quadratic kernel function.

The length_scale parameter $l$ controls the smoothness of the function and $\sigma^2$ controls the vertical variation.

Before applying Gaussian process, there's another function that I use it to compute the covariance matrix.
#### calculate covariance matrix
```python
def calculate_Covar(n, X, beta, var, alpha, length_scale):
    C = np.zeros(shape=(n, n), dtype=float)
    kernel = rational_quadratic_kernel_method(X, X, var, alpha, length_scale)
    C = kernel + (1 / beta) * np.eye(n)
    return C
```
$\begin{gather}C(X_n, X_m) = k(x_n, x_m) + \beta^{-1}\delta_{nm}\end{gather}$

$\begin{gather}\delta_{nm} = \begin{cases} 1,~n = m, \\ 0,~otherwise. \end{cases}\end{gather}$

Because the $\delta$ equals to 1 only when the n equals to m, instead the identity matrix is uesd to present the total $\delta_{nm}$.

Here we get the kernel function and the covariance matrix, so we could apply the two function to Gaussian Process.
#### Gaussian Proccess
```python
def Gaussian_Process(X, Y, X_pos, beta, var, alpha, length_scale):
    # compute the Covariance: (34, 34) 
    C = calculate_Covar(X.shape[0], X, beta, var, alpha, length_scale)

    # compute the similarity
    kernel_upR = rational_quadratic_kernel_method(X, X_pos, var, alpha, length_scale)
    kernel_downR = rational_quadratic_kernel_method(X_pos, X_pos, var, alpha, length_scale)
    kernel_downR += ( 1 / beta ) * np.eye(X_pos.shape[0])
    
    mean_point = kernel_upR.T @ inv(C) @ Y
    var_point = kernel_downR
    var_point -= kernel_upR.T @ inv(C) @ kernel_upR
    return mean_point, var_point
```
X: training data (34, 1)
Y: training label (34, 1)
X_pos: testing X (1000, 1)
var, alpha, length_scale: parameter of kernel function

Here X_pos's shape is (1000, 1) because we would like to predict the distribution from [-60, 60]. So we space 1000 number ranging from (-60, 60) and the detail would be further discuss in the main function.

As we already know the **marginal likelihood** would be like:
![](https://i.imgur.com/M5Ye3N0.png)

And the prediction would be:
![](https://i.imgur.com/X40tl9F.png)

Most importance of all, the conditional distribution $p(y^*|y)$ is Gaussian distribution with:
$\begin{gather}\mu(x^*)= k(x, x^*)^T C^{-1}y \end{gather}$
$\begin{gather}\sigma^2(x^*) = k^* - k(x, x^*)^T C^{-1} k(x. x^*) \end{gather}$
$\begin{gather}k^* = k(x^*, x^*) +\beta^{-1} \end{gather}$

So in the above code, we calculate the kernel_upR to measure the similarity of training data and testing data. The variable kernel_upR is same as $k(x, x^*)$. 
kernel_downR equals to the computation $k(x^*, x^*)+\beta^{-1}$

The shape of mean is (1000,1) and the shape of variance is (1000, 1000).

After computing these variable, then we could get the mean and variance and plot them.
#### Visualization
```python
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
    plt.title(text + "beta: {:.2f}, var: {:.2f}, alpha: {:.2f}, length_scale: {:.2f}".format(args[0], args[1], args[2], args[3]))
    plt.xlabel("X")
    plt.ylabel("Y")
```
For the part of visualization, we plot the $\mu$ line with the blue color. And the corresponding code is
```python
plt.plot(X_pos, mean_pos, "b")
```
Also, all the data points are also showed. 
```python
plt.scatter(X, Y, c="#ff00f2")
```
For the 95% confidence inteval of f, the Z value is 1.96. So the upper line is mean + 1.96 * $\sigma$ and the lower line is mean - 1.96 * $\sigma$ plotting with red color and filling color between these interval. 
```python
plt.plot(X_pos, mean_pos + 1.96 * standard, "r")
plt.plot(X_pos, mean_pos - 1.96 * standard, "r")

upper = mean_pos + 1.96 * standard
lower = mean_pos - 1.96 * standard
plt.fill_between(X_pos[:, 0], upper[:, 0], lower[:, 0], alpha = 0.2, color="#ffe294")
```

Then we go to the main function.
```python
# read the input.data  X: (34,1 ) Y: (34, 1) 
X, Y = read_input()

# Apply Gaussian Process Regression to predict the distribution of f
beta = 5
var = 1
alpha = 1
length_scale = 1

X_point = np.linspace(-60, 60, 1000).reshape(-1, 1)

# Gaussian Process with initial value 
mean_1, var_1 = Gaussian_Process(X, Y, X_point, beta, var, alpha, length_scale)           
plot_figure(X, Y, X_point, mean_1, var_1, [beta, var, alpha, length_scale])
```
Initialize the beta, var, alpha and length_scale and we put them to Gaussian Process function and plot them.



#### Optimize the kernel parameters - minimizing the negative marginal log-likelihood
```python
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
```
Here I implemented 2 optimization function, one is naive one and the other is more stable. But actually in this experiment, the more stable one doesn't have any significant difference from the naive one.
The stable cost function first use cholesky decomposion to get the L. Secondly, using the solve_triangular to solve the equation Lx = Y for x, x here equals to S1. Thirdly, L.Tx = S1, x here equals to S2. The det(C_opt) in the naive cost function is replaced by the np.diagonal(L) and the inv(C_opt) @ Y is replaced by the S2.

The marginal likelihood if function of $\theta$:
$\begin{gather}
p(y|\theta) = N(y|0, C_{\theta})\\
\ln p(y | \theta) = -{1 \over 2} \ln |C_{\theta}| -{1 \over 2}y^TC_{\theta}^{-1}y -{N \over 2} \ln(2 \pi)
\end{gather}$
$\theta$ : the set of var, alpha and length_scale.

Optimal values for these parameters can be estimated by minimizing the negative log likelihood w.r.t parameters var, alpha and length_scale.

Theta in costFunction is the kernel parameter that we want to find.

Then we use scipy.optimize.minimize to minimize the marginal log-likelihood.
```python
from scipy.optimize import minimize

result = minimize(costFunction_naive, [var, alpha, length_scale], args=(X, Y, beta), bounds=((1e-8, 1e6), (1e-4, 1e6), (1e-8, 1e6)) )
var_opt = result.x[0]
alpha_opt = result.x[1]
length_scale_opt = result.x[2]
```
Notice here, we need to set the bounds. Because all these variable need to be postive. Otherwise, these variable might be extreme and the result would be weird.

### Part b. experiments settings and results
The graph with initial value would be like that:
beta = 5, var = 1, alpha = 1, length_scale = 1
![](https://i.imgur.com/99A7X8h.png)

After getting the optimal value, the figure would be like that: (using the naive cost function)
beta = 5, var = 1.73, alpha = 250.60, length_scale = 3.31
![](https://i.imgur.com/x9GYLDt.png)

Using the stable one:
beta = 5, var = 1.73, alpha = 564.16, length_scale = 3.32
![](https://i.imgur.com/L4pCnud.png)

**Comparison:**
* dependent variable: length_scale $l$

| length_scale | figure                               |
| ------------ | ------------------------------------ |
| 0.1          | ![](https://i.imgur.com/sJP5khS.png) |
| 1            | ![](https://i.imgur.com/99A7X8h.png) |
| 10           | ![](https://i.imgur.com/Uw5zdQq.png) |

* dependent variable: alpha $\alpha$

| alpha | figure                               |
| ----- | ------------------------------------ |
| 0.01  | ![](https://i.imgur.com/tfDIRKK.png)|
| 0.1   | ![](https://i.imgur.com/Wf7VZ0L.png)|
| 1     | ![](https://i.imgur.com/99A7X8h.png)|
| 10    | ![](https://i.imgur.com/dnfajlg.png)|

* dependent variable: var $\sigma^2$

| var | figure                               |
| ----- | ------------------------------------ |
| 0.1  | ![](https://i.imgur.com/vfF3IcF.png)|
| 1   | ![](https://i.imgur.com/99A7X8h.png)  |
| 10     | ![](https://i.imgur.com/918Slrg.png)|


### Part c. observations and discussion
The rational quadratic kernel is equivalent to adding many SE kernels together with different length-scales.
1. length_scale $l$: smoothness
    * higher length_scale -> more smooth -> coarser approximations of the training data. (Figure Test 6 length_scale = 10)
    * lower length_scale -> more wiggly -> wide uncertainty regions between training data points. (Figure Test 3 length_scale = 0.1)

2. alpha $\alpha$: scale meansure parameter
    when $\alpha \to \infty$, the rational quadratic kernel is identical to SE (Squared Exponential Covariance Function).
4. var $\sigma^2$: vertical variation 
    * higher var -> wider in uncertain regions. (Figure Test 1 var=10 & Test 5 var=0.1)

## SVM 
### Part a: code with detailed explanation
#### I. different kernel functions
```python
import numpy as np
import time # calculate the compute time
from libsvm.svmutil import *

def openCsv(filePath):
    data = np.genfromtxt(filePath, delimiter=',')
    return data
    
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
```
First we load the data through the function *openCsv* using the numpy.
X_train shape: (5000, 784)
Y_train shape: (5000, 1)
X_test shape: (2500, 784)
Y_test shape: (2500, 1)

Document: [libsvm](https://github.com/cjlin1/libsvm/tree/eedbb147ea79af44f2ecdca1064f2c6a8fe6462d)

Follow the document, libsvm has provided multiple options for svm_train.

options:
-s svm_type : set type of SVM (default 0)
> 	0 -- C-SVC		(multi-class classification)
> 	1 -- nu-SVC		(multi-class classification)
> 	2 -- one-class SVM
> 	3 -- epsilon-SVR	(regression)
> 	4 -- nu-SVR		(regression)
    
-t kernel_type : set type of kernel function (default 2)
> 	0 -- linear: u'*v
> 	1 -- polynomial: (gamma*u'*v + coef0)^degree
> 	2 -- radial basis function: exp(-gamma*|u-v|^2)
> 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
> 	4 -- precomputed kernel (kernel values in training_set_file)

So in the above code, I use the dict to map different kernel function, linear equals to 0, polynomial equals to 1, and the radial basis function equals to 2. Then iterate the dict and train the model using different kernel function by changing the kernel type.

train & test:
```python
"""
svm_train(Y, X, command)
return svm.svm_model
"""
command = f"-t {kernel[k]}"
m = svm_train(Y_train, X_train, command)
p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
```
p_label: predicted labels
p_acc: accuracy
p_val: a 2D list, each row contains 10 kernel value. The column with higher value would be the result label.

#### II. Grid search
```python
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
                 
            if k == "polynomial":
                for g in gamma:
                    for d in degree:
                        for co in coef:
                          #print(k, "gamma:", g, "d", d, "coef0", co)
                          command = f"-t {kernel[k]} -c {c} -g {g} -d {d} -v 10 -r {co}"
                          opt_acc, opt_cmd = findBestParameter(Y_train, X_train, opt_acc, opt_cmd, command)
                          #print(k," opt_acc : ", opt_acc)
                          #print()
            if k == "RBF":
                for g in gamma:
                    #print(k, "cost:", c, "gamma:", g)
                    command = f"-t {kernel[k]} -c {c} -g {g} -v 10"
                    opt_acc, opt_cmd = findBestParameter(Y_train, X_train, opt_acc, opt_cmd, command)
                    #print(k," opt_acc : ", opt_acc)
                    #print()    
    return opt_acc, opt_cmd
    
opt_acc, opt_cmd= gridSearch(X_train, Y_train)
       
opt_model = svm_train(Y_train, X_train, opt_cmd)
opt_label, opt_test_acc, opt_val = svm_predict(Y_test, X_test, opt_model) 
```
There are 4 kernel that we frequently use:
1. linear $K(x, z) = x^Tz$
2. polynomial: $K(x, z) = (\gamma x^Tz+\gamma)^d, \gamma>0$
3. radial basis function(RBF): $K(x, z)= e^{-\gamma ||x-z||^ 2}$
4. sigmoid kernel: $K(x, z) = tanh(\gamma x^Tz + 4)$

for the linear:
there would be 3 kind of parameters that we should tune for.
1. $C$: the parameter C of C-SVC (default 1)

for the polynomial:
there would be 3 kind of parameters that we should tune for.
1. $C$: the parameter C of C-SVC (default 1)
2. $\gamma$: gamma (default 1/num_features)
3. $d$: degree (default 3)
4. $coef$: coef0 in kernel function (default 0)

for the radial basis function:
there would be 2 kind of parameters that we sould tune for.
1. $C$: the parameter C of C-SVC (default 1)
2. $\gamma$: gamma (default 1/num_features)

By default the SVM is C-SVC.

Following the document, we could tune the parameters by adding the options. 
> -d degree : set degree in kernel function (default 3)
> -g gamma : set gamma in kernel function (default 1/num_features)
> -r coef0 : set coef0 in kernel function (default 0)
> -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)

The most importance of all, because we would use the k-fold cross validation to find the best parameter. Here I use the 10-fold. In the code, we only need to add the options to command:

> -v n: n-fold cross validation mode

Notice that if we turn to the cross validation mode, the return value of svm_train would only be the accuracy of cross validation. The return value would be like that:
```
Cross Validation Accuracy = 96.8%
```
So here, the function findBestParameter is used to evaluate whether the new set of parameter is better than the current best one.

Then we use the grid search to iterate. After we get the best parameter set and the best command, we predict the test dataset.

#### III. linear + RBF kernel
```python
def linearKernel(u, v):
    return u @ v.T

def RBFKernel(u, v, gamma):
    dist = np.sum(u ** 2, axis=1).reshape(-1, 1) + np.sum(v **2, axis=1) - 2 * u @ v.T
    return np.exp(-gamma *  dist)
    
'''
linear kernel + rbf kernel together => a new kernel function

'''
combineKernel = linearKernel(X_train, X_train) + RBFKernel(X_train, X_train, 0.01)
combineKernel_test = linearKernel(X_test, X_train) + RBFKernel(X_test, X_train, 0.01)

X_kernel = np.hstack((np.arange(1, 5001).reshape(-1, 1), combineKernel))
X_kernel_s = np.hstack((np.arange(1, 2501).reshape(-1, 1), combineKernel_test))

model = svm_train(Y_train, X_kernel, "-t 4")
p_label, p_acc, p_val = svm_predict(Y_test, X_kernel_s, model)
```
As we already know the linear kernel function and the radial basis function, we mixed them together and see the performance.
* linear $K(x, z) = x^Tz$
```python
def linearKernel(u, v):
    return u @ v.T
```
* radial basis function(RBF): $K(x, z)= e^{-\gamma ||x-z||^ 2}$
```python
def RBFKernel(u, v, gamma):
    dist = np.sum(u ** 2, axis=1).reshape(-1, 1) + np.sum(v **2, axis=1) - 2 * u @ v.T
    return np.exp(-gamma *  dist)
```
Here the gamma is set to 0.01, because in the grid search I find when the gamma is 0.01, the overall accuarcy is higher.

Before training, we need to add a **serial number** to the kernel value that we already had computed.

In the document, the first column would be the serial number:
> If the linear kernel is used, we have the following new
>	training/testing sets:
>   15  0:1 1:4 2:6  3:1
>	45  0:2 1:6 2:18 3:0
>	25  0:3 1:1 2:0  3:1
>    
>	15  0:? 1:2 2:0  3:1

```python
X_kernel = np.hstack((np.arange(1, 5001).reshape(-1, 1), combineKernel))
X_kernel_s = np.hstack((np.arange(1, 2501).reshape(-1, 1), combineKernel_test))
```
X_kernel shape would be like (5001, 5000). 
X_kernels shape would be like (2501, 5000).
The serial number must start from 1, otherwist the program would throw error exception.

Also the option need to be change to "-t 4", which means precomputed kernel.

### Part b. experiments settings and results
#### I. Use different kernel functions (linear, polynomial, and RBF kernels) and have comparison between their performance.
|          | Linear | Polynomial | RBF    |
| -------- | ------ | ---------- | ------ |
| Accuarcy | 95.08% | 34.68%     | 95.32% |
| Runtime  | 3.65 s | 27.99 s    | 7.87 s |

#### II. Grid search

After the grid search:
**Linear kernel:**
| Cost  | Accuracy |
| ----- | -------- |
| 0.001 | 95.64%   |
| 0.01  | 97.06%   |
| 0.1   | 97%      |
| 1     | 96.28%   |
| 10    | 96.22%   |

**Polynomial**


| Cost: 0.001                          | Cost: 0.01                           | Cost:0.1                             |
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/Z7gLCzc.png) | ![](https://i.imgur.com/n0vzQzY.png) | ![](https://i.imgur.com/Vu9Nd2C.png) |
| Cost: 1                              | Cost: 10                             |                                      |
| ![](https://i.imgur.com/41aXsA1.png) | ![](https://i.imgur.com/6qL5Oez.png) |                                      |

**radial basis function**
![](https://i.imgur.com/N00AIiC.png)



#### III. Linear kernel + RBF kernel together
Linear Fixed Cost = 1
RBF: Fixed Cost = 1
|             | Linear | RBF    | New    |
| ----------- | ------ | ------ | ------ |
| gamma=1/784 | 96.28% | 96.44% | 95.08% |
| gamma=0.01  | 96.28% | 97.96% | 95.32% |


RBF: Fixed gamma = 1/784
New: Fixed gamma = 1/784
|             | Linear | RBF    | New    |
| ----------- | ------ | ------ | ------ |
| cost = 10   | 96.22% | 97.12% | 95.00% |
| cost = 1    | 96.28% | 96.44% | 95.08% |
| cost = 0.01 | 97.06% | 81.20% | 95.96% |

### Part c. observations and discussion
#### I.
Before tuning any parameters, the best accuracy is SVM with RBF kernel. And the worst accuracy is SVM with polynomial kernel.
Because the default parameter of polynomial like degree equals to 3, and the coef equals to 0, which seems to be the worst combination found in the grid search.
#### II
Due to the limitation of hareware, I tried to run the 5+240+20=265 times combination of different set of parameters, which cost me 4 to 5 hours.

The command with the best accuracy is "-t 2 -c 10 -g 0.01", which is SVM with RBF kernel. parameter Cost = 10, gamma = 0.01. The accuracy of cross validation is 98.36%, and the accuracy of prediction is 98.2%.

Actually the performance of polynomial after tuning is also great, which accuracy of cross validation is high to 98.26%. This observations is quite interesting because before tuning the parameters, the performance of polynomial is pretty bad in experiment I. Also even when the degree equals to 3, if other parameter is set properly, the performance of the set could sill be good. But overall when the cost up to 1 or even 10, almost every set of performance is up to 95%.
#### III
The performance of the combined kernel is quite confunsing. Before the experiment, I thought the performance might be between the accuracy of linear kernel and RBF kernel. But The combined one is much worst than the original one. The assumption is corret only when the cost=0.01, the RBF kernel performs pretty bad (81.2%). The accuracy of the combined kernel is 95.96% between the linear and RBF kernel.

But overall the combined kernel performance is not good. I think maybe next time I could try to mix the kernles with a portion like $p$ * linear + $1-p$ * RBF kernel. Maybe the performance would be greater or much reasonable.


#### Reference
[Github libsvm](https://github.com/cjlin1/libsvm/tree/eedbb147ea79af44f2ecdca1064f2c6a8fe6462d1)