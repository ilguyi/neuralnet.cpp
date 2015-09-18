/***********************************************************
 * Neural networks for multi-class classification
 * some typedef namespace
 *
 * 2015. 08. 25.
 * modified 2015. 08. 25.
 * by Il Gu Yi
***********************************************************/

#ifndef NEURAL_NETWORK_DEF_H
#define NEURAL_NETWORK_DEF_H

#include <armadillo>
using namespace std;


namespace nndef {


typedef arma::mat Weight;
typedef arma::vec Bias;
typedef arma::field<Weight> Weights;
typedef arma::field<Bias> Biases;
typedef arma::mat Matrix;
typedef arma::vec Vector;


typedef enum {
    //  using cross entropy cost function
    //  C = target * log activation + (1 - target) * log (1 - activation) 
    CrossEntropy,
    //  using quadratic cost function C = (target - activation)^2 / 2
    Quadratic,
} CostFunction_Type;


typedef enum {
    Sigmoid,
    Tanh,
    Softmax,
    ReLU,
} ActivationFuntion_Type;



}



#endif
