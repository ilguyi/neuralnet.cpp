# Neural network for multiclass classification
implement with backpropagation using both Boost and Armadillo library in C++

# Author
Il Gu Yi


# Usage
- execute: ./nn train.d test.d parameter.d
- argument 1: train data (optional including validation data)
- argument 2: test data
- argument 3: parameters data
- DATA: MNIST (partial data) 


# What program can do (version 0.1) (2015. 07. 12.)
- Multi hidden layer
- Validation data (extracting randomly from train data)
- Two cost function (cross entropy or quadratic error)
- Apply regularization (weight decay)
- Two sigmoid function type (binary or bipolar)


# Requirement
- I use the random number generator mt19937 from Boost library
for weights and bias initialization and stochastic gradient descent.
- I implement my program using Armadillo linear algebra library in C++
for various calculation based on matrix and vector.

