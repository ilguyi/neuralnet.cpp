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

    unsigned N = nn.GetN_train() + nn.GetN_valid() + nn.GetN_test();
//    unsigned N = nn.GetN_train() + nn.GetN_valid();
    unsigned dimension = nn.GetDimension();

    //  Reading training data
    DataFrame<unsigned> trainMNIST;
//  Usage: ReadDataFile(filename, N, dimension, header, target)
    trainMNIST.ReadDataFile(argv[1], N, dimension, true, true);
    cout << "Read training data complete" << endl;

    //  Set target data
    trainMNIST.SetTargetMatrix(nn.GetTargetClass(), "Binary");
    cout << "Target Matrix complete" << endl;


    //  Randomly extract validation data
//    DataFrame<unsigned> validMNIST;
//    trainMNIST.SplitValidationSet(validMNIST, nn.GetN_valid());
//    cout << "Split validation data complete" << endl;

    //  Randomly extract validation data and test data
    DataFrame<unsigned> validMNIST, testMNIST;
    trainMNIST.SplitValidTestSet(validMNIST, nn.GetN_valid(), testMNIST, nn.GetN_test());
    cout << "Split validation data and test data complete" << endl;


    //  Reading test data
//    DataFrame<unsigned> testMNIST;
//    unsigned N_test = nn.GetN_test();
//    testMNIST.ReadDataFile(argv[2], N_test, dimension, true, false);
//    cout << "Read test data complete" << endl;


    //  Set linear scaling each features
    DataFrame<double> trainMNIST_norm, validMNIST_norm, testMNIST_norm;
    trainMNIST.NormalizationEachFeatures(validMNIST, testMNIST, trainMNIST_norm, validMNIST_norm, testMNIST_norm);
    cout << "Normalization all data complete" << endl;
//    trainMNIST_norm.PrintData();
//    validMNIST_norm.PrintData();
//    testMNIST_norm.PrintData();



//  Execute NeuralNetworks
    nn.Initialize();

    for (unsigned iter=0; iter<1; iter++) {

//        nn.Training(trainMNIST_norm, iter);
//        nn.Training(trainMNIST_norm, validMNIST_norm, iter);
        nn.Training(trainMNIST_norm, validMNIST_norm, testMNIST_norm, iter);

//        arma::uvec predict(N_test);
//        predict.zeros();
//        nn.Test(testMNIST_norm, predict, iter);
    }


    return 0;
}

