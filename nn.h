/***********************************************************
 * Neural networks for multi-class classification
 *
 * 2015. 06. 11.
 * modified 2015. 09. 21.
 * by Il Gu Yi
***********************************************************/

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <boost/random.hpp>
#include <armadillo>
#include <string>
#include "nndef.h"
#include "nnlayer.h"
using namespace std;
using namespace df;
using namespace nndef;
using namespace layer;


namespace nn {


typedef struct NeuralNetworkParameters {
    NeuralNetworkParameters() :
        N_train(0), dimension(0),
        N_valid(0), N_test(0),
        n_class(1),
        learningRate(0.5),
        cost(CrossEntropy),
        regularization(0.0),
        momentum(0.0),
        minibatchSize(1),
        softmax(false),
        maxEpoch(100) {
            n_hiddens;
            n_hlayer = n_hiddens.size();
            target_class;
        };

    unsigned N_train;
    unsigned dimension;
    unsigned N_valid;
    unsigned N_test;
    arma::uvec n_hiddens;
    unsigned n_hlayer;
    arma::ivec target_class;
    unsigned n_class;
    double learningRate;
    CostFunction_Type cost;
    double regularization;
    double momentum;
    unsigned minibatchSize;
    bool softmax;
    unsigned maxEpoch;
} NNParameters;




class NeuralNetworks {
    public:
        NeuralNetworks();
        NeuralNetworks(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
            const arma::uvec& n_hiddens, const arma::ivec& target_class, const double& learningRate, const CostFunction_Type& cost, 
            const double& regularization, const double& momentum, const unsigned& minibatchSize, const bool& softmax,
            const unsigned& maxEpoch);

        void ReadParameters(const string& filename);
        void ParametersSetting(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
            const arma::uvec& n_hiddens, const arma::ivec& target_class, const double& learningRate, const CostFunction_Type& cost, 
            const double& regularization, const double& momentum, const unsigned& minibatchSize, const bool& softmax,
            const unsigned& maxEpoch);
        void PrintParameters() const;
        void WriteParameters(const string& filename) const;

        unsigned GetN_train() const;
        unsigned GetDimension() const;
        unsigned GetN_valid() const;
        unsigned GetN_test() const;
        unsigned GetN_class() const;
        arma::ivec GetTargetClass() const;

        void Initialize();
//        void Initialize(const Weights& weight_init, const Biases& bias_init);

        void PrintWeights() const;
        void PrintBiases() const;
        void PrintResults() const;
        void WriteWeights(const string& filename) const;
        void WriteBiases(const string& filename, const string& append) const;
        void WriteResults(const string& filename) const;


    private:
        void NamingFile(string& filename);
        void NamingFileStep(string& filename, const unsigned& step);
        

    public:
        template<typename dataType>
        void Training(df::DataFrame<dataType>& data, const unsigned& step);
        template<typename dataType>
        void Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, unsigned& step);
        template<typename dataType>
        void Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, df::DataFrame<dataType>& test, unsigned& step);

    private:
        template<typename dataType>
        void TrainingOneStep(df::DataFrame<dataType>& data);
        template<typename dataType>
        void TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid);
        template<typename dataType>
        void TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, df::DataFrame<dataType>& test);

        void MiniBathces(arma::field<Vector>& minibatch);
        void InitializeDeltaParameters();
        template<typename dataType>
        void TrainingMiniBatch(df::DataFrame<dataType>& data, const Vector& minibatch, double& error);
        template<typename dataType>
        void TrainingMiniBatch(df::DataFrame<dataType>& data, const Vector& minibatch, double& error, double& accuracy);


        template<typename dataType>
        void FeedForward(const arma::Row<dataType>& x);

        void CostFunction(double& error, const arma::ivec& t);

        template<typename dataType>
        void Validation(df::DataFrame<dataType>& valid, double& error, double& accuracy);
        template<typename dataType>
        void Test(df::DataFrame<dataType>& valid, double& error, double& accuracy);

        template<typename dataType>
        void BackPropagation(const arma::Row<dataType>& x, const arma::ivec& t);
        
        void UpdateParameter(const unsigned& minibatchSize);
        
        void WriteError(const string& filename, const double& error, const double& accuracy);
        void WriteError(const string& filename, const double& error, const double& accuracy, const double& valid_error, const double& valid_accuracy);
        void WriteError(const string& filename, const double& error, const double& accuracy,
                const double& valid_error, const double& valid_accuracy, const double& test_error, const double& test_accuracy);


    public:
        template<typename dataType>
        void Test(df::DataFrame<dataType>& data, arma::uvec& predict, unsigned& step);


    private:
        arma::field<Layer> layers;
        NNParameters nnParas;
};

NeuralNetworks::NeuralNetworks() {};
NeuralNetworks::NeuralNetworks(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
    const arma::uvec& n_hiddens, const arma::ivec& target_class, const double& learningRate, const CostFunction_Type& cost, 
    const double& regularization, const double& momentum, const unsigned& minibatchSize, const bool& softmax,
    const unsigned& maxEpoch) {

    nnParas.N_train = N_train;
    nnParas.dimension = dimension;
    nnParas.N_valid = N_valid;
    nnParas.N_test = N_test;
    nnParas.n_hiddens = n_hiddens;
    nnParas.target_class = target_class;
    nnParas.n_class = nnParas.target_class.size();
    nnParas.n_hlayer = nnParas.n_hiddens.size();
    nnParas.learningRate = learningRate;        
    nnParas.cost = cost;
    nnParas.regularization = regularization;
    nnParas.momentum = momentum;
    nnParas.minibatchSize = minibatchSize;
    nnParas.softmax = softmax;
    nnParas.maxEpoch = maxEpoch;
}


void NeuralNetworks::ReadParameters(const string& filename) {

    ifstream fin(filename.c_str());
    string s;
    for (unsigned i=0; i<4; i++) getline(fin, s);
    nnParas.N_train = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.dimension = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.N_valid = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.N_test = stoi(s);

    getline(fin, s);    getline(fin, s);
    stringstream ss(s);
    while ( getline(ss, s, ' ') ) {
        nnParas.n_hiddens.resize(nnParas.n_hiddens.size()+1);
        nnParas.n_hiddens(nnParas.n_hiddens.size()-1) = stoi(s);
    }
    nnParas.n_hlayer = nnParas.n_hiddens.size();

    getline(fin, s);    getline(fin, s);
    stringstream sss(s);
    while ( getline(sss, s, ' ') ) {
        nnParas.target_class.resize(nnParas.target_class.size()+1);
        nnParas.target_class(nnParas.target_class.size()-1) = stoi(s);
    }
    nnParas.n_class = nnParas.target_class.size();

    getline(fin, s);    getline(fin, s);
    nnParas.learningRate = stod(s);

    getline(fin, s);    getline(fin, s);    getline(fin, s);
    getline(fin, s);    getline(fin, s);
    if ( s == "CrossEntropy" ) nnParas.cost = CrossEntropy;
    else if ( s == "Quadratic" ) nnParas.cost = Quadratic;
    else {
        cout << "Usage: Wrong Cost function type" << endl;
        cout << "you must type CrossEntropy or Quadratic" << endl;
    }

    getline(fin, s);    getline(fin, s);
    nnParas.regularization = stod(s);
    
    getline(fin, s);    getline(fin, s);
    nnParas.momentum = stod(s);

    getline(fin, s);    getline(fin, s);
    nnParas.minibatchSize = stoi(s);

    getline(fin, s);    getline(fin, s);

    if ( s == "true"  ||  s == "T"  ||  s == "True" ) nnParas.softmax = true;
    else if ( s == "false"  ||  s == "F"  ||  s == "False" ) nnParas.softmax = false;
    else {
        cout << "Usage: Wrong Softmax boolean type" << endl;
        cout << "you must type true or false" << endl;
    }

    getline(fin, s);    getline(fin, s);
    nnParas.maxEpoch = stoi(s);
}


void NeuralNetworks::ParametersSetting(const unsigned& N_train, const unsigned& dimension, const unsigned& N_valid, const unsigned& N_test,
    const arma::uvec& n_hiddens, const arma::ivec& target_class, const double& learningRate, const CostFunction_Type& cost, 
    const double& regularization, const double& momentum, const unsigned& minibatchSize, const bool& softmax,
    const unsigned& maxEpoch) {

    nnParas.N_train = N_train;
    nnParas.dimension = dimension;
    nnParas.N_valid = N_valid;
    nnParas.N_test = N_test;
    nnParas.n_hiddens = n_hiddens;
    nnParas.target_class = target_class;
    nnParas.n_class = nnParas.target_class.size();
    nnParas.n_hlayer = nnParas.n_hiddens.size();
    nnParas.learningRate = learningRate;        
    nnParas.cost = cost;
    nnParas.regularization = regularization;
    nnParas.momentum = momentum;
    nnParas.minibatchSize = minibatchSize;
    nnParas.softmax = softmax;
    nnParas.maxEpoch = maxEpoch;
}

void NeuralNetworks::PrintParameters() const {
    cout << "##################################"    << endl;
    cout << "##  Neural Networks Parameters  ##"    << endl;
    cout << "##################################"    << endl << endl;
    cout << "Number of train data: "                << nnParas.N_train << endl;
    cout << "dimension: "                           << nnParas.dimension << endl;
    cout << "Number of validation data: "           << nnParas.N_valid << endl;
    cout << "Number of test data: "                 << nnParas.N_test << endl;

    for (unsigned i=0; i<nnParas.n_hiddens.size(); i++)
        cout << "number of nodes in " << i+1 << " hidden layer: " << nnParas.n_hiddens(i) << endl ;

    cout << "number of class: "                     << nnParas.n_class << endl ;
    cout << "number of hidden layer: "              << nnParas.n_hlayer << endl ;
    cout << "learning rate: "                       << nnParas.learningRate << endl;

    if ( nnParas.cost == CrossEntropy ) cout << "cost function: Cross Entropy" << endl;
    else cout << "cost function: Quadratic" << endl;

    cout << "regularization rate: "                 << nnParas.regularization << endl;
    cout << "momentum rate: "                       << nnParas.momentum << endl;
    cout << "minibatch size: "                      << nnParas.minibatchSize << endl;

    if ( nnParas.softmax ) cout << "use of softmax: Yes" << endl;
    else cout << "use of softmax: No" << endl;

    cout << "iteration max epochs: "                << nnParas.maxEpoch << endl << endl;
}

void NeuralNetworks::WriteParameters(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave << "##################################"   << endl;
    fsave << "##  Neural Networks Parameters  ##"   << endl;
    fsave << "##################################"   << endl << endl;
    fsave << "Number of train data: "               << nnParas.N_train << endl;
    fsave << "dimension: "                          << nnParas.dimension << endl;
    fsave << "Number of validation data: "          << nnParas.N_valid << endl;
    fsave << "Number of test data: "                << nnParas.N_test << endl;

    for (unsigned i=0; i<nnParas.n_hiddens.size(); i++)
        fsave << "number of nodes in " << i+1 << " hidden layer: " << nnParas.n_hiddens(i) << endl ;

    fsave << "number of class: "                    << nnParas.n_class << endl ;
    fsave << "number of hidden layer: "             << nnParas.n_hlayer << endl ;
    fsave << "learning_Rate: "                      << nnParas.learningRate << endl;

    if ( nnParas.cost == CrossEntropy ) fsave << "cost function: Cross Entropy" << endl;
    else fsave << "cost function: Quadratic" << endl;

    fsave << "regularization rate: "                << nnParas.regularization << endl;
    fsave << "momentum rate: "                      << nnParas.momentum << endl;
    fsave << "minibatch size: "                     << nnParas.minibatchSize << endl;

    if ( nnParas.softmax ) fsave << "use of softmax: Yes" << endl;
    else fsave << "use of softmax: No" << endl;

    fsave << "iteration max epochs: "               << nnParas.maxEpoch << endl << endl;
    fsave.close();
}


unsigned NeuralNetworks::GetN_train() const { return nnParas.N_train; }
unsigned NeuralNetworks::GetDimension() const { return nnParas.dimension; }
unsigned NeuralNetworks::GetN_valid() const { return nnParas.N_valid; }
unsigned NeuralNetworks::GetN_test() const { return nnParas.N_test; }
unsigned NeuralNetworks::GetN_class() const { return nnParas.n_class; }
arma::ivec NeuralNetworks::GetTargetClass() const { return nnParas.target_class; }


void NeuralNetworks::Initialize() {

    layers.set_size(nnParas.n_hlayer+1);

    layers(0).Initialize_Layer(nnParas.dimension, nnParas.n_hiddens(0), Sigmoid, 0.0);
    for (unsigned l=1; l<nnParas.n_hlayer; l++)
        layers(l).Initialize_Layer(nnParas.n_hiddens(l-1), nnParas.n_hiddens(l), Sigmoid, 0.0);
    layers(nnParas.n_hlayer).Initialize_Layer(nnParas.n_hiddens(nnParas.n_hlayer-1), nnParas.n_class, Softmax, 0.0);
}



void NeuralNetworks::PrintWeights() const {
//  cout.precision(10);
//  cout.setf(ios::fixed);

    layers(0).weight.raw_print("weights matrix between input and hidden 1");
    cout << endl;

    for (unsigned l=1; l<nnParas.n_hlayer; l++) {
        string ment = "weights matrix between hidden ";
        stringstream ss;    ss << l;
        ment += ss.str();
        ment += " and hidden ";
        ss.str("");
        ss << l + 1;
        ment += ss.str();

        layers(l).weight.raw_print(ment);
        cout << endl;
    }

    string ment = "weights matrix between hidden ";
    stringstream ss;    ss << nnParas.n_hlayer;
    ment += ss.str();
    ment += " and output";

    layers(nnParas.n_hlayer).weight.raw_print(ment);
    cout << endl;
}

void NeuralNetworks::PrintBiases() const {
//  cout.precision(10);
//  cout.setf(ios::fixed);

    for (unsigned l=0; l<nnParas.n_hlayer; l++) {
        string ment = "bias vector in hidden ";
        stringstream ss;    ss << l+1;
        ment += ss.str();

        layers(l).bias.raw_print(ment);
        cout << endl;
    }

    layers(nnParas.n_hlayer).bias.raw_print("bias vector in output");
    cout << endl;
}

void NeuralNetworks::PrintResults() const {
    PrintWeights();
    PrintBiases();
}


void NeuralNetworks::WriteWeights(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave.precision(10);
    fsave.setf(ios::fixed);

    layers(0).weight.raw_print(fsave, "weights matrix between input and hidden 1");
    fsave << endl;

    for (unsigned l=1; l<nnParas.n_hlayer; l++) {
        string ment = "weights matrix between hidden ";
        stringstream ss;    ss << l;
        ment += ss.str();
        ment += " and hidden ";
        ss.str("");
        ss << l + 1;
        ment += ss.str();

        layers(l).weight.raw_print(fsave, ment);
        fsave << endl;
    }

    string ment = "weights matrix between hidden ";
    stringstream ss;    ss << nnParas.n_hlayer;
    ment += ss.str();
    ment += " and output";

    layers(nnParas.n_hlayer).weight.raw_print(fsave, ment);
    fsave << endl;
}


void NeuralNetworks::WriteBiases(const string& filename, const string& append) const {
    ofstream fsave;
    if ( append != "append")
        fsave.open(filename.c_str());
    else
        fsave.open(filename.c_str(), fstream::out | fstream::app);
    fsave.precision(10);
    fsave.setf(ios::fixed);

    for (unsigned l=0; l<nnParas.n_hlayer; l++) {
        string ment = "bias vector in hidden ";
        stringstream ss;    ss << l+1;
        ment += ss.str();

        layers(l).bias.raw_print(fsave, ment);
        fsave << endl;
    }

    layers(nnParas.n_hlayer).bias.raw_print(fsave, "bias vector in output");
    fsave << endl;
}

void NeuralNetworks::WriteResults(const string& filename) const {
    WriteWeights(filename);
    WriteBiases(filename, "append");
}



void NeuralNetworks::NamingFile(string& filename) {
    stringstream ss;
    for (unsigned l=0; l<nnParas.n_hlayer; l++) {
        filename += "h";
        ss << nnParas.n_hiddens(l);
        filename += ss.str();    ss.str("");
    }
    filename += "lr";
    ss << nnParas.learningRate;
    filename += ss.str();    ss.str("");

    filename += "rg";
    ss << nnParas.regularization;
    filename += ss.str();    ss.str("");

    filename += "mo";
    ss << nnParas.momentum;
    filename += ss.str();    ss.str("");

    filename += ".txt";
}

void NeuralNetworks::NamingFileStep(string& filename, const unsigned& step) {
    stringstream ss;
    for (unsigned l=0; l<nnParas.n_hlayer; l++) {
        filename += "h";
        ss << nnParas.n_hiddens(l);
        filename += ss.str();    ss.str("");
    }
    filename += "lr";
    ss << nnParas.learningRate;
    filename += ss.str();    ss.str("");

    filename += "rg";
    ss << nnParas.regularization;
    filename += ss.str();    ss.str("");

    filename += "mo";
    ss << nnParas.momentum;
    filename += ss.str();    ss.str("");

    filename += "step";
    ss << step;
    filename += ss.str();    ss.str("");

    filename += ".txt";
}





template<typename dataType>
void NeuralNetworks::Training(df::DataFrame<dataType>& data, const unsigned& step) {

    string parafile = "nn.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned epoch=0; epoch<nnParas.maxEpoch; epoch++) {
        cout << "epochs: " << epoch << endl;
        double remain_epoch = (double) epoch / (double) nnParas.maxEpoch * 100;
        cout << "remaining epochs ratio: " << remain_epoch << "%" << endl << endl;
        TrainingOneStep(data);
    }
    
    if ( step % 10 == 0 ) {
        string resfile = "nn.result.";
        NamingFileStep(resfile, step);
        WriteResults(resfile);
    }
}

template<typename dataType>
void NeuralNetworks::Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, unsigned& step) {

    string parafile = "nn.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned epoch=0; epoch<nnParas.maxEpoch; epoch++) {
        cout << "epochs: " << epoch << endl;
        double remain_epoch = (double) epoch / (double) nnParas.maxEpoch * 100;
        cout << "remaining epochs ratio: " << remain_epoch << "%" << endl << endl;
        TrainingOneStep(data, valid);
    }
    
    if ( step % 10 == 0 ) {
        string resfile = "nn.result.";
        NamingFileStep(resfile, step);
        WriteResults(resfile);
    }
}

template<typename dataType>
void NeuralNetworks::Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, df::DataFrame<dataType>& test, unsigned& step) {

    string parafile = "nn.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned epoch=0; epoch<nnParas.maxEpoch; epoch++) {
        cout << "epochs: " << epoch << endl;
        double remain_epoch = (double) epoch / (double) nnParas.maxEpoch * 100;
        cout << "remaining epochs ratio: " << remain_epoch << "%" << endl << endl;
        TrainingOneStep(data, valid, test);
    }
    
    if ( step % 10 == 0 ) {
        string resfile = "nn.result.";
        NamingFileStep(resfile, step);
        WriteResults(resfile);
    }
}




template<typename dataType>
void NeuralNetworks::TrainingOneStep(df::DataFrame<dataType>& data) {

    arma::field<Vector> minibatch;
    MiniBathces(minibatch);

    double error = 0.0;
    double accuracy = 0.0;
    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), error, accuracy);
    error /= (double) nnParas.N_train;
    accuracy /= (double) nnParas.N_train;

    string errorfile = "nn.error.";
    NamingFile(errorfile);
    WriteError(errorfile, error, accuracy);
}

template<typename dataType>
void NeuralNetworks::TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid) {

    arma::field<Vector> minibatch;
    MiniBathces(minibatch);

    double error = 0.0;
    double accuracy = 0.0;
    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), error, accuracy);
    error /= (double) nnParas.N_train;
    accuracy /= (double) nnParas.N_train;


    //  validation test
    double valid_error = 0.0;
    double valid_accuracy = 0.0;
    Validation(valid, valid_error, valid_accuracy);


    string errorfile = "nn.error.";
    NamingFile(errorfile);
    WriteError(errorfile, error, accuracy, valid_error, valid_accuracy);
}

template<typename dataType>
void NeuralNetworks::TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, df::DataFrame<dataType>& test) {

    arma::field<Vector> minibatch;
    MiniBathces(minibatch);

    double error = 0.0;
    double accuracy = 0.0;
    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), error, accuracy);
    error /= (double) nnParas.N_train;
    accuracy /= (double) nnParas.N_train;


    //  validation test
    double valid_error = 0.0;
    double valid_accuracy = 0.0;
    Validation(valid, valid_error, valid_accuracy);

    //  validation test
    double test_error = 0.0;
    double test_accuracy = 0.0;
    Test(test, test_error, test_accuracy);

    string errorfile = "nn.error.";
    NamingFile(errorfile);
    WriteError(errorfile, error, accuracy, valid_error, valid_accuracy, test_error, test_accuracy);
}


void NeuralNetworks::MiniBathces(arma::field<Vector>& minibatch) {

    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);                 //  Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

    Vector rand_data(nnParas.N_train);
    for (unsigned n=0; n<nnParas.N_train; n++)
        rand_data(n) = urnd();
    arma::uvec shuffleindex = sort_index(rand_data);

    unsigned n_minibatch = (unsigned) (nnParas.N_train / nnParas.minibatchSize);
    unsigned remainder = nnParas.N_train % nnParas.minibatchSize;
    if ( remainder != 0 ) {
        n_minibatch++;
        minibatch.set_size(n_minibatch);
        unsigned index = 0;
        for (unsigned n=0; n<n_minibatch-1; n++) {
            minibatch(n).set_size(nnParas.minibatchSize);
            for (unsigned j=0; j<nnParas.minibatchSize; j++)
                minibatch(n)(j) = shuffleindex(index++);
        }
        minibatch(n_minibatch-1).set_size(remainder);
        for (unsigned j=0; j<remainder; j++)
            minibatch(n_minibatch-1)(j) = shuffleindex(index++);
    }
    else {
        minibatch.set_size(n_minibatch);
        unsigned index = 0;
        for (unsigned n=0; n<n_minibatch; n++) {
            minibatch(n).set_size(nnParas.minibatchSize);
            for (unsigned j=0; j<nnParas.minibatchSize; j++)
                minibatch(n)(j) = shuffleindex(index++);
        }
    }
}


void NeuralNetworks::InitializeDeltaParameters() {

    for (unsigned l=0; l<nnParas.n_hlayer+1; l++) {
        layers(l).delta_weight.zeros();
        layers(l).delta_bias.zeros();
    }
}


template<typename dataType>
void NeuralNetworks::TrainingMiniBatch(df::DataFrame<dataType>& data, const Vector& minibatch, double& error) {

    InitializeDeltaParameters();

    for (unsigned n=0; n<minibatch.size(); n++) {

        //  Pick one record
        arma::Row<dataType> x = data.GetDataRow(minibatch(n));
        arma::irowvec t = data.GetTargetMatrixRow(minibatch(n));

        //  FeedForward learning
        FeedForward(x);

        //  Error estimation
        CostFunction(error, t.t());

        //  Error Back Propagation
        BackPropagation(x, t.t());
    }

    //  Update Parameters
    UpdateParameter(minibatch.size());
}

template<typename dataType>
void NeuralNetworks::TrainingMiniBatch(df::DataFrame<dataType>& data, const Vector& minibatch, double& error, double& accuracy) {

    InitializeDeltaParameters();

    for (unsigned n=0; n<minibatch.size(); n++) {

        //  Pick one record
        arma::Row<dataType> x = data.GetDataRow(minibatch(n));
        arma::irowvec t = data.GetTargetMatrixRow(minibatch(n));

        //  FeedForward learning
        FeedForward(x);

        //  Error estimation
        CostFunction(error, t.t());

        //  accuracy estimation
        arma::uword index;
        double max_value = layers(nnParas.n_hlayer).activation.max(index);
        if ( nnParas.target_class(index) != data.GetTarget(minibatch(n)) ) accuracy += 1.;

        //  Error Back Propagation
        BackPropagation(x, t.t());
    }

    //  Update Parameters
    UpdateParameter(minibatch.size());
}



template<typename dataType>
void NeuralNetworks::FeedForward(const arma::Row<dataType>& x) {

    layers(0).Forward(x.t());
    for (unsigned l=1; l<nnParas.n_hlayer+1; l++)
        layers(l).Forward(layers(l-1).activation);
}


void NeuralNetworks::CostFunction(double& error, const arma::ivec& t) {
    if ( nnParas.cost != CrossEntropy )
        error += arma::dot(t - layers(nnParas.n_hlayer).activation, t - layers(nnParas.n_hlayer).activation) * 0.5;
    else
        error += - arma::dot(t, log(layers(nnParas.n_hlayer).activation)) -
            arma::dot(1. - t, log(1. - layers(nnParas.n_hlayer).activation));

    double w_sum = 0.;
    if ( nnParas.regularization != 0. ) {
        for (unsigned l=0; l<nnParas.n_hlayer+1; l++)
            w_sum += arma::accu(layers(l).weight % layers(l).weight);
        w_sum *= nnParas.regularization * .5;
    }
    error += w_sum;
}




template<typename dataType>
void NeuralNetworks::Validation(df::DataFrame<dataType>& valid, double& error, double& accuracy) {

    for (unsigned n=0; n<nnParas.N_valid; n++) {

        //  Pick one record
        arma::Row<dataType> x = valid.GetDataRow(n);
        arma::irowvec t = valid.GetTargetMatrixRow(n);

        //  FeedForward learning
        FeedForward(x);

        //  Error estimation
        CostFunction(error, t.t());

        //  accuracy estimation
        arma::uword index;
        double max_value = layers(nnParas.n_hlayer).activation.max(index);
        if ( nnParas.target_class(index) != valid.GetTarget(n) ) accuracy += 1.;
    }
    error /= (double) nnParas.N_valid;
    accuracy /= (double) nnParas.N_valid;
}

template<typename dataType>
void NeuralNetworks::Test(df::DataFrame<dataType>& test, double& error, double& accuracy) {

    for (unsigned n=0; n<nnParas.N_test; n++) {

        //  Pick one record
        arma::Row<dataType> x = test.GetDataRow(n);
        arma::irowvec t = test.GetTargetMatrixRow(n);

        //  FeedForward learning
        FeedForward(x);

        //  Error estimation
        CostFunction(error, t.t());

        //  accuracy estimation
        arma::uword index;
        double max_value = layers(nnParas.n_hlayer).activation.max(index);
        if ( nnParas.target_class(index) != test.GetTarget(n) ) accuracy += 1.;
    }
    error /= (double) nnParas.N_test;
    accuracy /= (double) nnParas.N_test;
}



template<typename dataType>
void NeuralNetworks::BackPropagation(const arma::Row<dataType>& x, const arma::ivec& t) {

    layers(nnParas.n_hlayer).Backward(t, nnParas.cost);
    for (unsigned l=nnParas.n_hlayer-1; l>0; l--)
        layers(l).Backward(layers(l+1));
    layers(0).Backward(layers(1));


//  Cumulation of delta_weight and delta_bias in minibatch
    layers(0).Cumulation(x);
    for (unsigned l=1; l<nnParas.n_hlayer+1; l++)
        layers(l).Cumulation(layers(l-1).activation.t());
}


void NeuralNetworks::UpdateParameter(const unsigned& minibatchSize) {

    for (unsigned l=0; l<nnParas.n_hlayer+1; l++)
        //  Without Momentum
        //layers(l).WithoutMomentum(nnParas.learningRate, nnParas.regularization, nnParas.N_train, minibatchSize);
        //  Momentum
        //layers(l).Momentum(nnParas.learningRate, nnParas.regularization, nnParas.momentum, nnParas.N_train, minibatchSize);
        //  Nesterov Momentum
        layers(l).NesterovMomentum(nnParas.learningRate, nnParas.regularization, nnParas.momentum,
                nnParas.N_train, minibatchSize);
}


void NeuralNetworks::WriteError(const string& filename, const double& error, const double& accuracy) {
    ofstream fsave(filename.c_str(), fstream::out | fstream::app);
    fsave << error << " " << accuracy << endl;
    fsave.close();
}

void NeuralNetworks::WriteError(const string& filename, const double& error, const double& accuracy, const double& valid_error, const double& valid_accuracy) {
    ofstream fsave(filename.c_str(), fstream::out | fstream::app);
    fsave << error << " " << accuracy << " " << valid_error << " " << valid_accuracy << endl;
    fsave.close();
}

void NeuralNetworks::WriteError(const string& filename, const double& error, const double& accuracy, 
        const double& valid_error, const double& valid_accuracy, const double& test_error, const double& test_accuracy) {
    ofstream fsave(filename.c_str(), fstream::out | fstream::app);
    fsave << error << " " << accuracy << " " 
        << valid_error << " " << valid_accuracy << " "
        << test_error << " " << test_accuracy << endl;
    fsave.close();
}



template<typename dataType>
void NeuralNetworks::Test(df::DataFrame<dataType>& data, arma::uvec& predict, unsigned& step) {

    for (unsigned n=0; n<data.GetN(); n++) {
        //  Pick one record
        arma::Row<dataType> x = data.GetDataRow(n);

        //  FeedForward learning
        FeedForward(x);

        arma::uword index;
        double max_value = layers(nnParas.n_hlayer).activation.max(index);
        predict(n) = index;
    }

    string predfile = "nn.predict.";
    NamingFileStep(predfile, step);

    ofstream fsave(predfile.c_str());
    predict.raw_print(fsave, "predeict test data");
    fsave.close();
}



}





#endif
