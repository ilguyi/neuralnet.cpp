/***********************************************************
 * Neural networks for multi-class classification namespace
 *
 * 2015. 06.
 * modified 2015. 07. 15.
 * by Il Gu Yi
***********************************************************/

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <boost/random.hpp>
#include <armadillo>
#include <string>
using namespace std;
using namespace df;


namespace nn {


typedef arma::field<arma::mat> Weights;
typedef arma::field<arma::vec> Biases;
typedef arma::vec Target;
typedef arma::mat Matrix;
typedef arma::vec Vector;



typedef enum {
    //  using cross entropy cost function
    //  C = target * log activation + (1 - target) * log (1 - activation) 
    CrossEntropy,
    //  using quadratic cost function C = (target - activation)^2 / 2
    Quadratic,
} CostFunction_Type;



typedef struct NeuralNetworkParameters {
    NeuralNetworkParameters() :
        N(0), dimension(0), N_valid(0),
        n_class(1),
        learningRate(0.5),
        cost(CrossEntropy),
        regularization(0.0),
        minibatchSize(1),
        softmax(false),
        shape_sigmoid(Binary),
        sigmoid_alpha(1.0),
        maxStep(100) {
            n_hiddens;
            n_hlayer = n_hiddens.size();
        };

    unsigned N;
    unsigned dimension;
    unsigned N_valid;
    arma::uvec n_hiddens;
    unsigned n_class;
    unsigned n_hlayer;
    double learningRate;
    CostFunction_Type cost;
    double regularization;
    unsigned minibatchSize;
    bool softmax;
    df::Sigmoid_Type shape_sigmoid;
    double sigmoid_alpha;
    unsigned maxStep;
} NNParameters;


class NeuralNetworks {
    public:
        NeuralNetworks();
        NeuralNetworks(const unsigned& N_, const unsigned& dimension_, const unsigned& N_valid_, const arma::uvec& n_hiddens_, const unsigned& n_class_,
            const double& learningRate_, const CostFunction_Type& cost_, const double& regularization_, const unsigned& minibatchSize,
            const bool& softmax_, const df::Sigmoid_Type& shape_sigmoid_, const double& sigmoid_alpha_, const unsigned& maxStep_);

        void ReadParameters(const string& filename);
        void ParametersSetting(const unsigned& N_, const unsigned& dimension_, const unsigned& N_valid_, const arma::uvec& n_hiddens_, const unsigned& n_class_,
            const double& learningRate_, const CostFunction_Type& cost_, const double& regularization_, const unsigned& minibatchSize,
            const bool& softmax_, const df::Sigmoid_Type& shape_sigmoid_, const double& sigmoid_alpha_, const unsigned& maxStep_);
        void PrintParameters() const;
        void WriteParameters(const string& filename) const;

        unsigned GetN() const;
        unsigned GetDimension() const;
        unsigned GetN_valid() const;
        unsigned GetOutput() const;
        df::Sigmoid_Type GetSigmoidType() const;

        void Initialize(const string& initialize_type);

    private:
        void Initialize_Uniform(Matrix& weight, Vector& bias);
        void Initialize_Gaussian(Matrix& weight, Vector& bias);

    public:
        void PrintWeights() const;
        void PrintBiases() const;
        void PrintResult() const;
        void WriteWeights(const string& filename) const;
        void WriteBiases(const string& filename, const string& append) const;
        void WriteResult(const string& filename) const;

//      void ReadResultFile(const string& filename);

    private:
        void NamingFile(string& filename);
        void NamingFileStep(string& filename, const unsigned& step);


    public:
        template<typename dataType>
        void Training(df::DataFrame<dataType>& data, const unsigned& step);
    private:
        template<typename dataType>
        void TrainingOneStep(df::DataFrame<dataType>& data);

    public:
        template<typename dataType>
        void Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, unsigned& step);
    private:
        template<typename dataType>
        void TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid);

        template<typename dataType>
        void Validation(df::DataFrame<dataType>& valid, double& error);


        void MiniBathces(arma::field<arma::vec>& minibatch);
        void InitializeDeltaParameters();
        template<typename dataType>
        void TrainingMiniBatch(df::DataFrame<dataType>& data, const Vector& minibatch, double& error);


        template<typename dataType>
        void FeedForward(const arma::Row<dataType>& x);

        void ActivationFunction(Vector& activation, Vector& summation);
        void SigmoidFuntion(Vector& activation, Vector& summation);
        Vector DerivativeSigmoid(Vector& x);
        void SoftmaxActivation(Vector& activation, Vector& summation);

        template<typename dataType>
        void BackPropagation(const arma::Row<dataType>& x, const arma::irowvec& t);
        
        void UpdateParameter(const unsigned& minibatchSize);
        
        void WriteError(const string& filename, const double& error);
        void WriteError(const string& filename, const double& error, const double& valid_error);


    public:
        template<typename dataType>
        void Test(df::DataFrame<dataType>& data, arma::uvec& predict, unsigned& step);


    private:
        Weights weight;
        Biases bias;

        arma::field<arma::vec> summation;
        arma::field<arma::vec> activation;

        arma::field<arma::vec> delta;
        Weights delta_weight;
        Biases delta_bias;

        NNParameters nnParas;
};

NeuralNetworks::NeuralNetworks() {};
NeuralNetworks::NeuralNetworks(const unsigned& N_, const unsigned& dimension_, const unsigned& N_valid_, const arma::uvec& n_hiddens_, const unsigned& n_class_,
    const double& learningRate_, const CostFunction_Type& cost_, const double& regularization_, const unsigned& minibatchSize_, const bool& softmax_,
    const df::Sigmoid_Type& shape_sigmoid_, const double& sigmoid_alpha_, const unsigned& maxStep_) {

    nnParas.N = N_;
    nnParas.dimension = dimension_;
    nnParas.N_valid = N_valid_;
    nnParas.n_hiddens = n_hiddens_;
    nnParas.n_class = n_class_;
    nnParas.n_hlayer = nnParas.n_hiddens.size();
    nnParas.learningRate = learningRate_;        
    nnParas.cost = cost_;
    nnParas.regularization = regularization_;
    nnParas.minibatchSize = minibatchSize_;
    nnParas.softmax = softmax_;
    nnParas.shape_sigmoid = shape_sigmoid_;
    nnParas.sigmoid_alpha = sigmoid_alpha_;
    nnParas.maxStep = maxStep_;
}



void NeuralNetworks::ReadParameters(const string& filename) {

    ifstream fin(filename.c_str());
    string s;
    for (unsigned i=0; i<4; i++) getline(fin, s);
    nnParas.N = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.dimension = stoi(s);

    getline(fin, s);    getline(fin, s);
    nnParas.N_valid = stoi(s);

    getline(fin, s);    getline(fin, s);
    stringstream ss(s);
    while ( getline(ss, s, ' ') ) {
        nnParas.n_hiddens.resize(nnParas.n_hiddens.size()+1);
        nnParas.n_hiddens(nnParas.n_hiddens.size()-1) = stoi(s);
    }
    nnParas.n_hlayer = nnParas.n_hiddens.size();

    getline(fin, s);    getline(fin, s);
    nnParas.n_class = stoi(s);

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
    nnParas.minibatchSize = stoi(s);

    getline(fin, s);    getline(fin, s);

    if ( s == "true"  ||  s == "T"  ||  s == "True" ) nnParas.softmax = true;
    else if ( s == "false"  ||  s == "F"  ||  s == "False" ) nnParas.softmax = false;
    else {
        cout << "Usage: Wrong Softmax boolean type" << endl;
        cout << "you must type true or false" << endl;
    }

    getline(fin, s);    getline(fin, s);
    if ( s == "Binary" ) nnParas.shape_sigmoid = Binary;
    else if ( s == "Bipolar" ) nnParas.shape_sigmoid = Bipolar;
    else {
        cout << "Usage: Wrong Sigmoid function type" << endl;
        cout << "you must type Binary or Bipolar" << endl;
    }

    getline(fin, s);    getline(fin, s);
    nnParas.sigmoid_alpha = stod(s);

    getline(fin, s);    getline(fin, s);
    nnParas.maxStep = stoi(s);
}



void NeuralNetworks::ParametersSetting(const unsigned& N_, const unsigned& dimension_, const unsigned& N_valid_, const arma::uvec& n_hiddens_, const unsigned& n_class_,
    const double& learningRate_, const CostFunction_Type& cost_, const double& regularization_, const unsigned& minibatchSize_, const bool& softmax_,
    const df::Sigmoid_Type& shape_sigmoid_, const double& sigmoid_alpha_, const unsigned& maxStep_) {

    nnParas.N = N_;
    nnParas.dimension = dimension_;
    nnParas.N_valid = N_valid_;
    nnParas.n_hiddens = n_hiddens_;
    nnParas.n_class = n_class_;
    nnParas.n_hlayer = nnParas.n_hiddens.size();
    nnParas.learningRate = learningRate_;        
    nnParas.cost = cost_;
    nnParas.regularization = regularization_;
    nnParas.minibatchSize = minibatchSize_;
    nnParas.softmax = softmax_;
    nnParas.shape_sigmoid = shape_sigmoid_;
    nnParas.sigmoid_alpha = sigmoid_alpha_;
    nnParas.maxStep = maxStep_;
}

void NeuralNetworks::PrintParameters() const {
    cout << "##################################"    << endl;
    cout << "##  Neural Networks Parameters  ##"    << endl;
    cout << "##################################"    << endl << endl;
    cout << "N: "                                   << nnParas.N << endl;
    cout << "dimension: "                           << nnParas.dimension << endl;
    cout << "Number of validation data: "           << nnParas.N_valid << endl;

    for (unsigned i=0; i<nnParas.n_hiddens.size(); i++)
        cout << "number of nodes in " << i+1 << " hidden layer: " << nnParas.n_hiddens(i) << endl ;

    cout << "number of class: "                     << nnParas.n_class << endl ;
    cout << "number of hidden layer: "              << nnParas.n_hlayer << endl ;
    cout << "learning rate: "                       << nnParas.learningRate << endl;

    if ( nnParas.cost == CrossEntropy ) cout << "cost function: Cross Entropy" << endl;
    else cout << "cost function: Quadratic" << endl;

    cout << "regularization rate: "                 << nnParas.regularization << endl;
    cout << "minibatch size: "                      << nnParas.minibatchSize << endl;

    if ( nnParas.softmax ) cout << "use of softmax: Yes" << endl;
    else cout << "use of softmax: No" << endl;

    if ( nnParas.shape_sigmoid == Binary ) cout << "sigmoid type: Binary" << endl;
    else cout << "sigmoid type: Bipolar" << endl;

    cout << "sigmoid alpha: "                       << nnParas.sigmoid_alpha << endl;
    cout << "max iteration step: "                  << nnParas.maxStep << endl << endl;
}

void NeuralNetworks::WriteParameters(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave << "##################################"   << endl;
    fsave << "##  Neural Networks Parameters  ##"   << endl;
    fsave << "##################################"   << endl << endl;
    fsave << "N: "                                  << nnParas.N << endl;
    fsave << "dimension: "                          << nnParas.dimension << endl;
    fsave << "Number of validation data: "          << nnParas.N_valid << endl;

    for (unsigned i=0; i<nnParas.n_hiddens.size(); i++)
        fsave << "number of nodes in " << i+1 << " hidden layer: " << nnParas.n_hiddens(i) << endl ;

    fsave << "number of class: "                    << nnParas.n_class << endl ;
    fsave << "number of hidden layer: "             << nnParas.n_hlayer << endl ;
    fsave << "learning_Rate: "                      << nnParas.learningRate << endl;

    if ( nnParas.cost == CrossEntropy ) fsave << "cost function: Cross Entropy" << endl;
    else fsave << "cost function: Quadratic" << endl;

    fsave << "regularization rate: "                << nnParas.regularization << endl;
    fsave << "minibatch size: "                     << nnParas.minibatchSize << endl;

    if ( nnParas.softmax ) fsave << "use of softmax: Yes" << endl;
    else fsave << "use of softmax: No" << endl;

    if ( nnParas.shape_sigmoid == Binary ) fsave << "sigmoid type: Binary" << endl;
    else fsave << "sigmoid type: Bipolar" << endl;

    fsave << "sigmoid alpha: "                      << nnParas.sigmoid_alpha << endl;
    fsave << "max iteration step: "                 << nnParas.maxStep << endl << endl;
    fsave.close();
}


unsigned NeuralNetworks::GetN() const { return nnParas.N; }
unsigned NeuralNetworks::GetDimension() const { return nnParas.dimension; }
unsigned NeuralNetworks::GetN_valid() const { return nnParas.N_valid; }
unsigned NeuralNetworks::GetOutput() const { return nnParas.n_class; }
df::Sigmoid_Type NeuralNetworks::GetSigmoidType() const { return nnParas.shape_sigmoid; }



void NeuralNetworks::Initialize(const string& initialize_type) {

    weight.set_size(nnParas.n_hlayer+1);
    bias.set_size(nnParas.n_hlayer+1);
    summation.set_size(nnParas.n_hlayer+1);
    activation.set_size(nnParas.n_hlayer+1);
    delta.set_size(nnParas.n_hlayer+1);
    delta_weight.set_size(nnParas.n_hlayer+1);
    delta_bias.set_size(nnParas.n_hlayer+1);

    weight(0).set_size(nnParas.n_hiddens(0), nnParas.dimension);
    bias(0).set_size(nnParas.n_hiddens(0));
    summation(0).set_size(nnParas.n_hiddens(0));
    activation(0).set_size(nnParas.n_hiddens(0));
    delta(0).set_size(nnParas.n_hiddens(0));
    delta_weight(0).set_size(nnParas.n_hiddens(0), nnParas.dimension);
    delta_bias(0).set_size(nnParas.n_hiddens(0));
    
    for (unsigned l=1; l<nnParas.n_hlayer; l++) {
        weight(l).set_size(nnParas.n_hiddens(l), nnParas.n_hiddens(l-1));
        bias(l).set_size(nnParas.n_hiddens(l));
        summation(l).set_size(nnParas.n_hiddens(l));
        activation(l).set_size(nnParas.n_hiddens(l));
        delta(l).set_size(nnParas.n_hiddens(l));
        delta_weight(l).set_size(nnParas.n_hiddens(l), nnParas.n_hiddens(l-1));
        delta_bias(l).set_size(nnParas.n_hiddens(l));
    }

    weight(nnParas.n_hlayer).set_size(nnParas.n_class, nnParas.n_hiddens(nnParas.n_hlayer-1));
    bias(nnParas.n_hlayer).set_size(nnParas.n_class);
    summation(nnParas.n_hlayer).set_size(nnParas.n_class);
    activation(nnParas.n_hlayer).set_size(nnParas.n_class);
    delta(nnParas.n_hlayer).set_size(nnParas.n_class);
    delta_weight(nnParas.n_hlayer).set_size(nnParas.n_class, nnParas.n_hiddens(nnParas.n_hlayer-1));
    delta_bias(nnParas.n_hlayer).set_size(nnParas.n_class);


    if ( initialize_type == "uniform" ) {
        for (unsigned l=0; l<nnParas.n_hlayer+1; l++)
            Initialize_Uniform(weight(l), bias(l));
    }
    else if ( initialize_type == "gaussian" ) {
        for (unsigned l=0; l<nnParas.n_hlayer+1; l++)
            Initialize_Gaussian(weight(l), bias(l));
    }
    else {
        cout << "Usage: you have to type {\"uniform\", \"gaussian\"}" << endl;
    }
}


void NeuralNetworks::Initialize_Uniform(Matrix& weight, Vector& bias) {

    double minmax = 1.0 / sqrt(weight.n_cols);
    boost::random::uniform_real_distribution<> uniform_real_dist(-minmax, minmax);      //  Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

    for (unsigned i=0; i<weight.n_rows; i++) {
        bias(i) = urnd();
        for (unsigned j=0; j<weight.n_cols; j++)
            weight(i, j) = urnd();
    }
}


void NeuralNetworks::Initialize_Gaussian(Matrix& weight, Vector& bias) {

    double std_dev = 1.0 / sqrt(weight.n_cols);
    boost::random::normal_distribution<> normal_dist(0.0, std_dev);         //  Choose a distribution
    boost::random::variate_generator<boost::random::mt19937 &,
        boost::random::normal_distribution<> > nrnd(rng, normal_dist);      //  link the Generator to the distribution

    for (unsigned i=0; i<weight.n_rows; i++) {
        bias(i) = nrnd();
        for (unsigned j=0; j<weight.n_cols; j++)
            weight(i, j) = nrnd();
    }
}

void NeuralNetworks::PrintWeights() const {
//  cout.precision(10);
//  cout.setf(ios::fixed);

    weight(0).raw_print("weights matrix between input and hidden 1");
    cout << endl;

    for (unsigned l=1; l<nnParas.n_hlayer; l++) {
        string ment = "weights matrix between hidden ";
        stringstream ss;    ss << l;
        ment += ss.str();
        ment += " and hidden ";
        ss.str("");
        ss << l + 1;
        ment += ss.str();

        weight(l).raw_print(ment);
        cout << endl;
    }

    string ment = "weights matrix between hidden ";
    stringstream ss;    ss << nnParas.n_hlayer;
    ment += ss.str();
    ment += " and output";

    weight(nnParas.n_hlayer).raw_print(ment);
    cout << endl;
}

void NeuralNetworks::PrintBiases() const {
//  cout.precision(10);
//  cout.setf(ios::fixed);

    for (unsigned l=0; l<nnParas.n_hlayer; l++) {
        string ment = "bias vector in hidden ";
        stringstream ss;    ss << l+1;
        ment += ss.str();

        bias(l).raw_print(ment);
        cout << endl;
    }

    bias(nnParas.n_hlayer).raw_print("bias vector in output");
    cout << endl;
}

void NeuralNetworks::PrintResult() const {
    PrintWeights();
    PrintBiases();
}


void NeuralNetworks::WriteWeights(const string& filename) const {
    ofstream fsave(filename.c_str());
    fsave.precision(10);
    fsave.setf(ios::fixed);

    weight(0).raw_print(fsave, "weights matrix between input and hidden 1");
    fsave << endl;

    for (unsigned l=1; l<nnParas.n_hlayer; l++) {
        string ment = "weights matrix between hidden ";
        stringstream ss;    ss << l;
        ment += ss.str();
        ment += " and hidden ";
        ss.str("");
        ss << l + 1;
        ment += ss.str();

        weight(l).raw_print(fsave, ment);
        fsave << endl;
    }

    string ment = "weights matrix between hidden ";
    stringstream ss;    ss << nnParas.n_hlayer;
    ment += ss.str();
    ment += " and output";

    weight(nnParas.n_hlayer).raw_print(fsave, ment);
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

        bias(l).raw_print(fsave, ment);
        fsave << endl;
    }

    bias(nnParas.n_hlayer).raw_print(fsave, "bias vector in output");
    fsave << endl;
}

void NeuralNetworks::WriteResult(const string& filename) const {
    WriteWeights(filename);
    WriteBiases(filename, "append");
}




/*
void NeuralNetworks::ReadResultFile(const string& filename) {

    weight_u.set_size(nnParas.dimension+1, nnParas.n_hidden);
    weight_v.set_size(nnParas.n_hidden+1, nnParas.n_class);

    ifstream fin(filename);
    string dum;
    getline(fin, dum);
    double value;
    for (unsigned i=0; i<nnParas.dimension+1; i++) {
        for (unsigned j=0; j<nnParas.n_hidden; j++) {
            fin >> value;
            weight_u(i, j) = value;
        }
    }

    fin >> dum >> dum >> dum >> dum >> dum >> dum;
    for (unsigned i=0; i<nnParas.n_hidden+1; i++) {
        for (unsigned j=0; j<nnParas.n_class; j++) {
            fin >> value;
            weight_v(i, j) = value;
        }
    }
}
*/


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

    for (unsigned iter=0; iter<nnParas.maxStep; iter++) {
        cout << "iteration: " << iter << endl;
        double remain_iter = (double) iter / (double) nnParas.maxStep * 100;
        cout << "remaining iteration ratio: " << remain_iter << "%" << endl << endl;
        TrainingOneStep(data);
    }
    
    string resfile = "nn.result.";
    NamingFileStep(resfile, step);
    WriteResult(resfile);
}


template<typename dataType>
void NeuralNetworks::TrainingOneStep(df::DataFrame<dataType>& data) {

    arma::field<arma::vec> minibatch;
    MiniBathces(minibatch);

    double error = 0.0;

    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), error);

    error /= (double) nnParas.N;

    string errorfile = "nn.error.";
    NamingFile(errorfile);
    WriteError(errorfile, error);
}




template<typename dataType>
void NeuralNetworks::Training(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid, unsigned& step) {

    string parafile = "nn.parameter.";
    NamingFile(parafile);
    WriteParameters(parafile);

    for (unsigned iter=0; iter<nnParas.maxStep; iter++) {
        cout << "iteration: " << iter << endl;
        double remain_iter = (double) iter / (double) nnParas.maxStep * 100;
        cout << "remaining iteration ratio: " << remain_iter << "%" << endl << endl;
        TrainingOneStep(data, valid);
    }
    
    string resfile = "nn.result.";
    NamingFileStep(resfile, step);
    WriteResult(resfile);
}


template<typename dataType>
void NeuralNetworks::TrainingOneStep(df::DataFrame<dataType>& data, df::DataFrame<dataType>& valid) {

    arma::field<arma::vec> minibatch;
    MiniBathces(minibatch);

    double error = 0.0;

    for (unsigned n=0; n<minibatch.size(); n++)
        TrainingMiniBatch(data, minibatch(n), error);

    error /= (double) nnParas.N;


    //  validation test
    double valid_error = 0.0;
    Validation(valid, valid_error);


    string errorfile = "nn.error.";
    NamingFile(errorfile);
    WriteError(errorfile, error, valid_error);
}


template<typename dataType>
void NeuralNetworks::Validation(df::DataFrame<dataType>& valid, double& error) {

    unsigned N = valid.GetN();

    for (unsigned n=0; n<N; n++) {
        //  Pick one record
        arma::Row<dataType> x = valid.GetDataRow(n);

        //  FeedForward learning
        FeedForward(x);

        arma::uword index;
        double max_value = activation(nnParas.n_hlayer).max(index);
        if ( index != valid.GetTarget(n) ) error += 1.0 / (double) N;
    }
}





void NeuralNetworks::MiniBathces(arma::field<arma::vec>& minibatch) {

    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);                 //  Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);      //  link the Generator to the distribution

    arma::vec rand_data(nnParas.N);
    for (unsigned n=0; n<nnParas.N; n++)
        rand_data(n) = urnd();
    arma::uvec shuffleindex = sort_index(rand_data);

    unsigned n_minibatch = (unsigned) (nnParas.N / nnParas.minibatchSize);
    unsigned remainder = nnParas.N % nnParas.minibatchSize;
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
        delta_weight(l).zeros();
        delta_bias(l).zeros();
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
        if ( nnParas.cost == CrossEntropy )
            error += - arma::dot(t.t(), log(activation(nnParas.n_hlayer))) -
                arma::dot(1.0 - t.t(), log(1.0 - activation(nnParas.n_hlayer)));
        else
            error += arma::dot(t.t() - activation(nnParas.n_hlayer), t.t() - activation(nnParas.n_hlayer)) * 0.5;

        //  Error Back Propagation
        BackPropagation(x, t);
    }

    //  Update Parameters
    UpdateParameter(minibatch.size());
}



template<typename dataType>
void NeuralNetworks::FeedForward(const arma::Row<dataType>& x) {

    summation(0) = weight(0) * x.t() + bias(0);
    ActivationFunction(activation(0), summation(0));

    for (unsigned i=1; i<nnParas.n_hlayer; i++) {
        summation(i) = weight(i) * activation(i-1) + bias(i);
        ActivationFunction(activation(i), summation(i));
    }

    summation(nnParas.n_hlayer) = weight(nnParas.n_hlayer) * activation(nnParas.n_hlayer-1) + bias(nnParas.n_hlayer);
    if ( nnParas.softmax )
        SoftmaxActivation(activation(nnParas.n_hlayer), summation(nnParas.n_hlayer));
    else
        ActivationFunction(activation(nnParas.n_hlayer), summation(nnParas.n_hlayer));



//  cout << summation << endl;
//  cout << activation << endl;
}


void NeuralNetworks::ActivationFunction(Vector& activation, Vector& summation) {
    SigmoidFuntion(activation, summation);
}

void NeuralNetworks::SigmoidFuntion(Vector& activation, Vector& summation) {
    if ( nnParas.shape_sigmoid == Binary )
        for (unsigned i=0; i<summation.size(); i++)
            activation(i) = 1.0 / (1.0 + exp(-nnParas.sigmoid_alpha * summation(i)));
    else
        for (unsigned i=0; i<summation.size(); i++)
            activation(i) = 2.0 / (1.0 + exp(-nnParas.sigmoid_alpha * summation(i))) - 1.0;
}

Vector NeuralNetworks::DerivativeSigmoid(Vector& x) {
    Vector temp;
    temp.set_size(x.size());
    SigmoidFuntion(temp, x);

    if ( nnParas.shape_sigmoid == Binary )
        temp = nnParas.sigmoid_alpha * temp % (1.0 - temp);
    else
        temp = nnParas.sigmoid_alpha * (1.0 + temp) % (1.0 - temp) * 0.5;

    return temp;
}



void NeuralNetworks::SoftmaxActivation(Vector& activation, Vector& summation) {

    activation = exp(summation) / arma::sum(exp(summation));
}




template<typename dataType>
void NeuralNetworks::BackPropagation(const arma::Row<dataType>& x, const arma::irowvec& t) {

    if ( nnParas.cost == Quadratic ) {
        delta(nnParas.n_hlayer) = (activation(nnParas.n_hlayer) - t.t()) % DerivativeSigmoid(summation(nnParas.n_hlayer));
        for (unsigned i=nnParas.n_hlayer-1; i>0; i--)
            delta(i) = (delta(i+1).t() * weight(i+1)).t() % DerivativeSigmoid(summation(i));
        delta(0) = (delta(1).t() * weight(1)).t() % DerivativeSigmoid(summation(0));
    }
    else {
        delta(nnParas.n_hlayer) = (activation(nnParas.n_hlayer) - t.t());
        for (unsigned i=nnParas.n_hlayer-1; i>0; i--)
            delta(i) = (delta(i+1).t() * weight(i+1)).t() % DerivativeSigmoid(summation(i));
        delta(0) = (delta(1).t() * weight(1)).t() % DerivativeSigmoid(summation(0));
    }


//  Cumulation of delta_weight and delta_bias in minibatch
    delta_weight(0) += delta(0) * x;
    delta_bias(0) += delta(0);

    for (unsigned l=1; l<nnParas.n_hlayer+1; l++) {
        delta_weight(l) += delta(l) * activation(l-1).t();
        delta_bias(l) += delta(l);
    }
}


void NeuralNetworks::UpdateParameter(const unsigned& minibatchSize) {

    for (unsigned l=0; l<nnParas.n_hlayer+1; l++) {
        weight(l) *= (1.0 - nnParas.regularization * nnParas.learningRate / (double) nnParas.N);
        weight(l) -= nnParas.learningRate * delta_weight(l) / (double) minibatchSize;
        bias(l) -= nnParas.learningRate * delta_bias(l) / (double) minibatchSize;
    }
}


void NeuralNetworks::WriteError(const string& filename, const double& error) {
    ofstream fsave(filename.c_str(), fstream::out | fstream::app);
    fsave << error << endl;
    fsave.close();
}

void NeuralNetworks::WriteError(const string& filename, const double& error, const double& valid_error) {
    ofstream fsave(filename.c_str(), fstream::out | fstream::app);
    fsave << error << " " << valid_error << endl;
    fsave.close();
}



template<typename dataType>
void NeuralNetworks::Test(df::DataFrame<dataType>& data, arma::uvec& predict, unsigned& step) {

    unsigned N = data.GetN();

    for (unsigned n=0; n<N; n++) {
        //  Pick one record
        arma::Row<dataType> x = data.GetDataRow(n);

        //  FeedForward learning
        FeedForward(x);

        arma::uword index;
        double max_value = activation(nnParas.n_hlayer).max(index);
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
