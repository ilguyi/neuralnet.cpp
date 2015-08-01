# Neural network for multi-class classification
implement with backpropagation using both Boost and Armadillo library in C++

# Author
Il Gu Yi


# Usage
- execute: ./nn train.d test.d parameter.d
- argument 1: train data (optional including validation data)
- argument 2: test data
- argument 3: parameters data
- DATA: MNIST (partial data) 


# What program can do (version 0.3) (2015. 08. 01.)
- Multiple hidden layer
- Validation data (extracting randomly from train data)
- Two cost function (cross entropy or quadratic error)
- Apply L2-regularization (weight decay)
- Adjust mini-batch size
- Two sigmoid function type (binary or bipolar)
- Softmax activation at output layer
- Apply momemtum parameter


# Requirement
- I use the random number generator mt19937 from Boost library
for weights and bias initialization and stochastic gradient descent.
- I implement my program using Armadillo linear algebra library in C++
for various calculation based on matrix and vector.


# Version 0.3 (2015. 08. 01.)
- Apply momemtum parameter

# Version 0.2 (2015. 07. 13.)
- Softmax activation at output layer

# Version 0.1 (2015. 07. 12.)
- Multiple hidden layer
- Validation data (extracting randomly from train data)
- Two cost function (cross entropy or quadratic error)
- Apply L2-regularization (weight decay)
- Adjust mini-batch size
- Two sigmoid function type (binary or bipolar)


