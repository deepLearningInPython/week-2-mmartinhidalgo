import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import minimize  # Python version of R's optim() function
from sklearn import datasets
 
# Carry out the exercises in your own copy of the notebook that you can find at
#    https://www.kaggle.com/code/datasniffer/perceptrons-mlp-s-and-gradient-descent.
# Then copy and paste code asked for below in between the dashed lines.
# Do not import additional packages.
 
# Task 1:
# Instructions:
# In the notebook, you wrote a function that implements an MLP with 2 hidden layers.
# The function should accept a vector of weights and a matrix X that stores input feature
# vectors in its **columns**.
# The name of the function should be my_mlp.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
# Setting up common test parameters
np.random.seed(223)
w = np.random.normal(size=(6*4 + 4*7 + 7))  # vector with input weight values
X = np.random.normal(size=(6,10))  # matrix with 6 feature values for each of 10 simulated observations

def my_mlp(w, X, sigma=np.tanh):
    #Make weights per layer
    W1 = np.array(w[0:(4*6)]).reshape(4, 6)
    W2 = np.array(w[(4*6):(7*4) + 6*4]).reshape(7, 4)
    W3 = np.array(w[7*4+4*6:]).reshape(1, 7)
    #Connect Xi to H1j with matrix multiplication
    a1 = sigma(W1 @ X)
    #Connect H1j to H2j with matrix multiplication
    a2 = sigma(W2 @ a1)
    #Connect H2j to f(X) with matrix multiplication
    f = sigma(W3 @ a2)
    return f

  # Test 1: Check output shape
def test_output_shape():
    output = my_mlp(w, X)
    assert output.shape == (1, 10), f"Expected shape (1, 10), but got {output.shape}"

# Test 2: Check output range for tanh
def test_output_range_tanh():
    output = my_mlp(w, X, sigma=np.tanh)
    assert np.all(output >= -1) and np.all(output <= 1), "Output is outside the expected tanh range of [-1, 1]"

# Test 3: Check output for a different activation function (e.g., sigmoid)
def test_output_sigmoid():
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    output = my_mlp(w, X, sigma=sigmoid)
    assert np.all(output >= 0) and np.all(output <= 1), "Output is outside the expected sigmoid range of [0, 1]"

# Test 4: Check output output for wrong handling of input
def test_output_value():
    output = np.round(my_mlp(w, X),7)
    expected_output = np.array([[-0.9985057 , -0.98895476,  0.98381408,  0.85878257, -0.98004358, -0.99413097, -0.99719557,  0.9290378 ,  0.66210054,  0.38851955]])
    expected_output = np.round(expected_output,7)
    assert np.max(abs(output - expected_output)) < 1e-9, "Failed on output values"
    
test_output_shape()
test_output_range_tanh()
test_output_sigmoid()
test_output_value()

# -----------------------------------------------
 
# Task 2:
# Instructions:
# In the notebook, you wrote a function that implements a loss function for training
# the MLP implemented by my_mlp of Task 1.
# The function should accept a vector of weights, a matrix X that stores input feature
# vectors in its **columns**, and a vector y that stores the target labels (-1 or +1).
# The name of the function should be MSE_func.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def MSE_func(w, X, y):
    #w = input weight values, X = matrix with feature values, y = observed
    f = my_mlp(w, X) # = predicted values (output of the network)
    MSE = np.sum((f - y)**2)
    return MSE
  
# Setting up common test parameters
np.random.seed(223)
w = np.random.normal(size=(6*4 + 4*7 + 7))  # vector with input weight values
X = np.random.normal(size=(6, 10))  # matrix with 6 feature values for each of 10 simulated observations
y = np.array([1 if i < X.shape[1] // 2 else -1 for i in range(X.shape[1])])  # target labels

# Test 1: Check output type
def test_mse_output_type():
    mse = MSE_func(w, X, y)
    assert isinstance(mse, (float, np.floating)), f"Expected output type float, but got {type(mse)}"

# Test 2: Check non-negative MSE
def test_mse_non_negative():
    mse = MSE_func(w, X, y)
    assert mse >= 0, f"Expected non-negative MSE, but got {mse}"

# Test 3: Check MSE with uniform weights and features
def test_mse_uniform_input():
    w_uniform = np.ones(shape=(6*4 + 4*7 + 7))
    X_uniform = np.ones(shape=(6, 10))
    mse = MSE_func(w_uniform, X_uniform, y)
    assert abs(mse-20) < 1e-4, f"Wrong value: {mse}"
    w_rev = w
    X_uniform = X/20
    mse = MSE_func(w_rev, X_uniform, y)
    assert abs(mse-12.220975600119619) < 1e-8, f"Wrong value (2): {mse}"
    
test_mse_output_type()
test_mse_non_negative()
test_mse_uniform_input()

# -----------------------------------------------
 
# Task 3:
# Instructions:
# In the notebook, you wrote a function that returns the gradient vector for the least
# squares (simple) linear regression loss function.
# The function should accept a vector beta that contains the intercept (β₀) and the slope (β₁),
# a vector x that stores values of the independent variable, and a vector y that stores
# the values of the dependent variable and should return an np.array() that has the derivative values
# as its components.
# The name of the function should be dR.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def dR(beta, x, y):
    dbeta_0 = 2*np.mean((beta[0] + beta[1]*x - y))   # implement the above formula for dR/dβ₀
    dbeta_1 = 2*np.mean((beta[0] + beta[1]*x - y)*x) # implement the above formula for dR/dβ₁
    return np.array([dbeta_0, dbeta_1])
  
  # Setting up common test parameters
beta = np.array([-0.1, 1.1])

# Test 1: Check output shape
def test_dR_output_shape():
    x = np.arange(-3, 3)
    y = np.arange(-3, 3) + 5
    d = dR(beta, x, y)
    assert d.shape == (2,), f"Expected shape (2,), but got {d.shape}"

# Test 2: Check output with all zeros for x and y
def test_dR_zeros_input():
    x = np.zeros(6)
    y = np.zeros(6)
    d = dR(beta, x, y)
    expected = np.array([2*beta[0], 0.0])  # Since x and y are zeros, derivatives should be zero
    assert np.allclose(d, expected), f"Expected {expected}, but got {d}"

# Test 3: Check dR with non-linear x values (squared values)
def test_dR_non_linear_x():
    x = np.arange(-3, 3)**2
    y = np.arange(-3, 3) + 5
    d = dR(beta, x, y)
    assert d.shape == (2,), f"Expected shape (2,), but got {d}"
    assert abs(d[0]+2.3-0.2/3) < 1e-7, f"Wrong value error: {d[0]}"
    
test_dR_output_shape()
test_dR_zeros_input()
test_dR_non_linear_x()
 
# -----------------------------------------------
