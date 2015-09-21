/***********************************************************
 * Neural networks for multi-class classification
 * Layer class
 *
 * 2015. 08. 25.
 * modified 2015. 09. 21.
 * by Il Gu Yi
***********************************************************/

#ifndef NN_LAYER_H
#define NN_LAYER_H

#include <boost/random.hpp>
#include <armadillo>
#include <string>
#include "nndef.h"
using namespace std;
using namespace df;
using namespace nndef;


namespace layer {


class Layer {
    public:
        Layer();
        
        void Initialize_Layer(const unsigned& n_input, const unsigned& n_output, const ActivationFuntion_Type& actFunc, const double& dropout);

        void Forward(const Vector& input);
        void Activation();

        void Backward(const arma::ivec& t, const CostFunction_Type& cost);
        void Backward(const Layer& nextLayer);

        Vector DerivativeSigmoid();
        Vector DerivativeTanh();
        Vector DerivativeReLU();

        template<typename dataType>
        void Cumulation(const arma::Row<dataType>& x);
        void Cumulation(const arma::rowvec& x);

        void WithoutMomentum(const double& learningRate, const double& regularization,
                const unsigned& N_train, const unsigned& minibatchSize);
        void Momentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);
        void NesterovMomentum(const double& learningRate, const double& regularization,
                const double& momentum, const unsigned& N_train, const unsigned& minibatchSize);



    public:
        Weight weight;
        Bias bias;

        Vector summation;
        Vector activation;

        Vector delta;
        Weight delta_weight;
        Bias delta_bias;

        Weight velocity_weight;
        Bias velocity_bias;

        ActivationFuntion_Type actFunc;
        double dropout;
};

Layer::Layer() {};

void Layer::Initialize_Layer(const unsigned& n_input, const unsigned& n_output, const ActivationFuntion_Type& actFunc, const double& dropout) {
    weight.set_size(n_output, n_input);
    bias.set_size(n_output);

    summation.set_size(n_output);
    activation.set_size(n_output);

    delta.set_size(n_output);
    delta_weight.set_size(n_output, n_input);
    delta_bias.set_size(n_output);

    velocity_weight.set_size(n_output, n_input);
    velocity_bias.set_size(n_output);

    this->actFunc = actFunc;
    if ( dropout >= 1.  ||  dropout < 0. ) cout << "Wrong dropout percent " << endl;
    this->dropout = dropout;

    //  weight Initialize from Gaussian distribution
    double std_dev = 1. / sqrt(weight.n_cols);
    boost::random::normal_distribution<> normal_dist(0., std_dev);          //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd(rng, normal_dist);      //  link the Generator to the distribution

    for (unsigned j=0; j<weight.n_cols; j++)
        for (unsigned i=0; i<weight.n_rows; i++)
            weight(i, j) = nrnd();

    boost::random::normal_distribution<> normal_dist1(0., 1.);              //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd1(rng, normal_dist1);    //  link the Generator to the distribution

    for (unsigned i=0; i<weight.n_rows; i++)
        bias(i) = nrnd1();

    velocity_weight.zeros();
    velocity_bias.zeros();
}


void Layer::Forward(const Vector& input) {
    summation = weight * input + bias;
    Activation();

/*    cout << "Forward input" << endl;
    cout << input << endl;
    cout << "Forward summation" << endl;
    cout << summation << endl;
    cout << "Forward activation" << endl;
    cout << activation<< endl;
    */

    if ( dropout != 0 ) {
        boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);      //  Choose a distribution
        boost::random::variate_generator<boost::mt19937 &,
            boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

        for (unsigned i=0; i<activation.size(); i++) {
            if ( urnd() < dropout ) activation(i) = 0.;
            else activation(i) *= 1./(1. - dropout);
        }
    }
}

void Layer::Activation() {
    
    if ( actFunc == Sigmoid )
        activation = 1. / (1. + exp(-summation));
    else if ( actFunc == Tanh )
        activation = tanh(summation);
    else if ( actFunc == Softmax )
        activation = exp(summation) / arma::sum(exp(summation));
    else if ( actFunc == ReLU )
        for (unsigned i=0; i<summation.size(); i++)
            activation(i) = summation(i) > 0 ? summation(i) : 0.;
}



void Layer::Backward(const arma::ivec& t, const CostFunction_Type& cost) {

    if ( cost != CrossEntropy ) {
        if ( actFunc == Sigmoid )
            delta = (activation - t) % DerivativeSigmoid();
        else if ( actFunc == Tanh )
            delta = (activation - t) % DerivativeTanh();
        else if ( actFunc == ReLU )
            delta = (activation - t) % DerivativeReLU();
    }
    else {
        delta = (activation - t);
    }

/*    cout << "Backward target" << endl;
    cout << t << endl;
    cout << "Backward delta" << endl;
    cout << delta << endl;
    */
}


void Layer::Backward(const Layer& nextLayer) {

    if ( actFunc == Sigmoid )
        delta = (nextLayer.delta.t() * nextLayer.weight).t() % DerivativeSigmoid();
    else if ( actFunc == Tanh )
        delta = (nextLayer.delta.t() * nextLayer.weight).t() % DerivativeTanh();
    else if ( actFunc == ReLU )
        delta = (nextLayer.delta.t() * nextLayer.weight).t() % DerivativeReLU();

/*    cout << "Backward nextLayer delta" << endl;
    cout << nextLayer.delta << endl;
    cout << "Backward nextLayer weight" << endl;
    cout << nextLayer.weight << endl;
    cout << "Backward delta" << endl;
    cout << delta << endl;

    cout << "Backward ********" << endl;
    cout << (nextLayer.delta.t() * nextLayer.weight).t() << endl;
    cout << "Backward ********" << endl;
    cout << DerivativeSigmoid() << endl;
    */
}



Vector Layer::DerivativeSigmoid() {
    return activation % (1. - activation);
}

Vector Layer::DerivativeTanh() {
    return (1. + activation) % (1. - activation) * 0.5;
}

Vector Layer::DerivativeReLU() {
    Vector temp(summation.size());
    for (unsigned i=0; i<summation.size(); i++)
        temp(i) = summation(i) > 0 ? 1. : 0.; 

    return temp;
} 


template<typename dataType>
void Layer::Cumulation(const arma::Row<dataType>& x) {
    delta_weight += delta * x;
    delta_bias += delta;

/*    cout << "Cumulation delta_weight" << endl;
    cout << delta_weight << endl;
    cout << "Cumulation delta_bias" << endl;
    cout << delta_bia << endl;
    */
}

void Layer::Cumulation(const arma::rowvec& x) {
    delta_weight += delta * x;
    delta_bias += delta;

/*    cout << "Cumulation delta_weight" << endl;
    cout << delta_weight << endl;
    cout << "Cumulation delta_bias" << endl;
    cout << delta_bias << endl;
    */
}


void Layer::WithoutMomentum(const double& learningRate, const double& regularization, const unsigned& N_train, const unsigned& minibatchSize) {

    this->weight *= (1. - regularization * learningRate / (double) N_train);
    this->weight -= learningRate * this->delta_weight / (double) minibatchSize;
    this->bias -= learningRate * this->delta_bias / (double) minibatchSize;
}

void Layer::Momentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    this->velocity_weight = momentum * this->velocity_weight
            - this->weight * regularization * learningRate / (double) N_train
            - learningRate * this->delta_weight / (double) minibatchSize;
    this->weight += this->velocity_weight;

    this->velocity_bias = momentum * this->velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += this->velocity_bias;
}

void Layer::NesterovMomentum(const double& learningRate, const double& regularization,
        const double& momentum, const unsigned& N_train, const unsigned& minibatchSize) {

    Weight velocity_weight_prev = this->velocity_weight;
    Bias velocity_bias_prev = this->velocity_bias;

    this->velocity_weight = momentum * this->velocity_weight - this->weight * regularization * learningRate / (double) N_train
        - learningRate * this->delta_weight / (double) minibatchSize;
    this->weight += (1. + momentum) * this->velocity_weight - momentum * velocity_weight_prev;

    this->velocity_bias = momentum * this->velocity_bias - learningRate * this->delta_bias / (double) minibatchSize;
    this->bias += (1. + momentum) * this->velocity_bias - momentum * velocity_bias_prev;
}



}


#endif
