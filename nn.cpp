#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <boost/random.hpp>
#include <armadillo>
#include "dataframe.h"
#include "nn.h"
using namespace std;
using namespace df;
using namespace nn;



int main (int argc, char** argv)
{
    if ( argc != 4 ) {
        cout << "Usage: nn training_data test_data parameters_data" << endl;
        exit(1);
    }

    //  Reading Parameters file
    NeuralNetworks nn;
    nn.ReadParameters(argv[3]);
    nn.PrintParameters();

    unsigned N = nn.GetN() + nn.GetN_valid();
    unsigned dimension = nn.GetDimension();



    //  Reading training data
    DataFrame<unsigned> MNIST;
//  Usage: ReadDataFile(filename, N, dimension, header, target)
    MNIST.ReadDataFile(argv[1], N, dimension, "True", "True");
    cout << "Read training data complete" << endl;
//  MNIST.PrintData();

    //  Set target data
    arma::uvec target_class;
    target_class << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9;
    if ( target_class.size() != nn.GetOutput() ) {
        cout << "Usage: number of class is NOT the same" << endl;
        exit(1);
    }
    df::Sigmoid_Type shape_sigmoid = nn.GetSigmoidType();
    MNIST.SetTargetMatrix(target_class, shape_sigmoid);
    cout << "Target Matrix complete" << endl;
//  MNIST.PrintTarget();
//  MNIST.PrintTargetMatrix();

    //  Set linear scaling each features
    DataFrame<double> normMNIST;
    MNIST.LinearScalingEachFeatures(normMNIST);
    cout << "Normalize training data complete" << endl;
//  normMNIST.PrintData();


    //  Randomly extract validation data
    DataFrame<double> validMNIST;
    normMNIST.SplitValidationSet(validMNIST, nn.GetN_valid());
    cout << "Split validation data complete" << endl;




    //  Reading test data
    DataFrame<unsigned> MNIST_test;
    unsigned N_test = 30;
    MNIST_test.ReadDataFile(argv[2], N_test, dimension, "True", "False");
    cout << "Read test data complete" << endl;
//  MNIST_test.PrintData();

    //  Set linear scaling each features
    DataFrame<double> normMNIST_test;
    MNIST_test.LinearScalingEachFeatures(normMNIST_test);
    cout << "Normalize test data complete" << endl;
//  normMNIST_test.PrintData();



//  Execute NeuralNetworks
    nn.Initialize("gaussian");


    for (unsigned iter=0; iter<10; iter++) {

//      nn.Training(normMNIST, iter);
        nn.Training(normMNIST, validMNIST, iter);

        arma::uvec predict(N_test);
        predict.zeros();

        nn.Test(normMNIST_test, predict, iter);
    }




    return 0;
}


