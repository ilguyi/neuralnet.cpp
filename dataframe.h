/***********************************************************
 * Data frame for machine learning (NN, RBM, DBN, etc)
 *
 * 2015. 06.
 * modified 2015. 09. 18.
 * by Il Gu Yi
***********************************************************/

#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <boost/random.hpp>
#include <armadillo>
using namespace std;

namespace df {


//  global variable rng for simplicity
boost::random::mt19937 rng(time(0));    //  Pick the Random Number Generator method



template<typename dataType>
class DataFrame {
    public:
        DataFrame();
        DataFrame(const unsigned& n, const unsigned& dim, const bool& isT);
        void ReadDataFile(const string& filename, const unsigned& N, const unsigned& dimension, const bool& header, const bool& boolTarget);
        void PrintData() const;
        void PrintTarget() const;
        void PrintTargetMatrix() const;

        unsigned GetN() const;
        unsigned GetDimension() const;
        bool IsTarget() const;

        arma::Mat<dataType> GetData() const;
        arma::ivec GetTarget() const;
        arma::imat GetTargetMatrix() const;
 
        dataType GetData(const unsigned& i, const unsigned& j) const;
        int GetTarget(const unsigned& i) const;
        int GetTargetMatrix(const unsigned& i, const unsigned& j) const;

        void SetN(const unsigned& N);
        void SetDimension(const unsigned& dim);
        void SetIsTarget(const bool& isTarget);
        void SetDataSize(const unsigned& N, const unsigned& dim);

        void SetSubset(const arma::Mat<dataType>& data, const arma::ivec& target, const arma::imat targetM, const arma::uvec& validindex);

        void SetData(dataType& value, const unsigned& i, const unsigned& j);
        void SetTargetMatrix(arma::ivec& target_class, const string& shape);

        arma::Row<dataType> GetDataRow(const unsigned& i) const;
        arma::Col<dataType> GetDataCol(const unsigned& j) const;
        void SwapRowsData(const unsigned& i, const unsigned& j);
        void SwapColsData(const unsigned& i, const unsigned& j);

        arma::irowvec GetTargetMatrixRow(const unsigned& i) const;
        arma::ivec GetTargetMatrixCol(const unsigned& j) const;
        void CopyTarget(const arma::ivec& target);
        void CopyTargetMatrix(const arma::imat& targetM);

        void LinearScalingEachFeatures(DataFrame<double>& x);
        void LinearScalingEachFeatures(DataFrame<dataType>& v, DataFrame<dataType>& t, DataFrame<double>& x, DataFrame<double>& y, DataFrame<double>& z);
        void NormalizationEachFeatures(DataFrame<double>& x);
        void NormalizationEachFeatures(DataFrame<dataType>& v, DataFrame<dataType>& t, DataFrame<double>& x, DataFrame<double>& y, DataFrame<double>& z);

        void TransformBinaryData();
        void SplitValidationSet(DataFrame<dataType>& valid, const unsigned& n_valid);
        void SplitValidTestSet(DataFrame<dataType>& valid, const unsigned& n_valid, DataFrame<dataType>& test, const unsigned& n_test);


    private:
        arma::Mat<dataType> data;
        arma::ivec target;
        arma::imat targetMatrix;
        unsigned N;                 //  data size
        unsigned dimension;         //  data dimension
        bool isTarget;              //  whether classification data or clustering data
};
template<typename dataType>
DataFrame<dataType>::DataFrame() {};
template<typename dataType>
DataFrame<dataType>::DataFrame(const unsigned& n, const unsigned& dim, const bool& isT) :
    N(n), dimension(dim), isTarget(isT) {
    if ( isTarget == true ) {
        data.set_size(N, dimension);
        target.set_size(N);
    }
    else {
        data.set_size(N, dimension);
    }
}
template<typename dataType>
void DataFrame<dataType>::ReadDataFile(const string& filename, const unsigned& N, const unsigned& dimension,
    const bool& header, const bool& boolTarget) {
    this->N = N;
    this->dimension = dimension;
    data.set_size(this->N, this->dimension);
    target.set_size(this->N);

    ifstream fin(filename.c_str());
    dataType value;
    int targetvalue;
    if ( boolTarget == true ) {
        isTarget = true;
        if ( header == true ) {
            string dum;
            getline(fin, dum);
            for (unsigned i=0; i<this->N; i++) {
                fin >> targetvalue;
                target(i) = targetvalue;
                for (unsigned j=0; j<this->dimension; j++) {
                    fin >> value;
                    data(i, j) = value;
                }
            }
        }
        else {
            for (unsigned i=0; i<this->N; i++) {
                fin >> targetvalue;
                target(i) = targetvalue;
                for (unsigned j=0; j<this->dimension; j++) {
                    fin >> value;
                    data(i, j) = value;
                }
            }
        }
    }
    else {
        isTarget = false;
        if ( header == true ) {
            string dum;
            getline(fin, dum);
            for (unsigned i=0; i<this->N; i++) {
                for (unsigned j=0; j<this->dimension; j++) {
                    fin >> value;
                    data(i, j) = value;
                }
            }
        }
        else {
            for (unsigned i=0; i<this->N; i++) {
                for (unsigned j=0; j<this->dimension; j++) {
                    fin >> value;
                    data(i, j) = value;
                }
            }
        }
    }

    fin.close();
}



template<typename dataType>
void DataFrame<dataType>::PrintData() const {
//  cout.precision(6);
//  cout.setf(ios::fixed);
    data.raw_print("Print Data");
    cout << endl;
}

template<typename dataType>
void DataFrame<dataType>::PrintTarget() const {
    if ( !isTarget )
        cout << "Usage: This dataframe doesn't have target data" << endl;
    else {
        target.raw_print("Print target data");
        cout << endl;
    }
}

template<typename dataType>
void DataFrame<dataType>::PrintTargetMatrix() const {
    if ( !isTarget )
        cout << "Usage: This dataframe doesn't have target matrix data" << endl;
    else if ( !targetMatrix.size() )
        cout << "Usage: Target matrix is not activated!!" << endl << "       You must SetTargetMatrix() function!!" << endl << endl;
    else {
        targetMatrix.raw_print("Print Target matrix");
        cout << endl;
    }
}



template<typename dataType>
unsigned DataFrame<dataType>::GetN() const { return N; }
template<typename dataType>
unsigned DataFrame<dataType>::GetDimension() const { return dimension; }
template<typename dataType>
bool DataFrame<dataType>::IsTarget() const { return isTarget; }


template<typename dataType>
arma::Mat<dataType> DataFrame<dataType>::GetData() const { return data; }
template<typename dataType>
arma::ivec DataFrame<dataType>::GetTarget() const { return target; }
template<typename dataType>
arma::imat DataFrame<dataType>::GetTargetMatrix() const { return targetMatrix; }
 

template<typename dataType>
dataType DataFrame<dataType>::GetData(const unsigned& i, const unsigned& j) const { return data(i, j); }
template<typename dataType>
int DataFrame<dataType>::GetTarget(const unsigned& i) const { return target(i); }
template<typename dataType>
int DataFrame<dataType>::GetTargetMatrix(const unsigned& i, const unsigned& j) const { return targetMatrix(i, j); }


template<typename dataType>
void DataFrame<dataType>::SetN(const unsigned& N) { this->N = N; }
template<typename dataType>
void DataFrame<dataType>::SetDimension(const unsigned& dim) { dimension = dim; }
template<typename dataType>
void DataFrame<dataType>::SetIsTarget(const bool& isTarget) {
    if ( isTarget ) this->isTarget = true;
    else this->isTarget = false;
}
template<typename dataType>
void DataFrame<dataType>::SetDataSize(const unsigned& N, const unsigned& dim) { data.set_size(N, dim); }

template<typename dataType>
void DataFrame<dataType>::SetSubset(const arma::Mat<dataType>& data, const arma::ivec& target, const arma::imat targetM, const arma::uvec& index) {
    this->data = data.rows(index);
    this->target = target(index);
    this->targetMatrix = targetM.rows(index);
}




//template<typename dataType>
//void DataFrame<dataType>::SetTargetColumn(const unsigned& t) {
//    for (unsigned i=t; i>0; i--)
//        data.swap_cols(i-1, i);
//}



template<typename dataType>
void DataFrame<dataType>::SetData(dataType& value, const unsigned& i, const unsigned& j) { data(i, j) = value; }
template<typename dataType>
void DataFrame<dataType>::SetTargetMatrix(arma::ivec& target_class, const string& shape) {
    unsigned n_target = target_class.n_rows;
    targetMatrix.set_size(N, n_target);

    int true_class;
    int false_class;
    if ( shape == "Binary" ) {
        true_class = 1;        false_class = 0;
    }
    else if ( shape == "Bipolar" ) {
        true_class = 1;        false_class = -1;
    }
    else {
        cout << "Wrong shape_sigmoid argument" << endl;
        cout << "You should write \"Binary\" or \"Bipolar\"" << endl;
        exit(1);
    }

    for (unsigned i=0; i<N; i++) {
        for (unsigned j=0; j<n_target; j++) {
            if ( target_class[j] != target(i) )
                targetMatrix(i, j) = false_class;
            else
                targetMatrix(i, j) = true_class;
        }
    }
}




template<typename dataType>
arma::Row<dataType> DataFrame<dataType>::GetDataRow(const unsigned& i) const { return data.row(i); }
template<typename dataType>
arma::Col<dataType> DataFrame<dataType>::GetDataCol(const unsigned& j) const { return data.col(j); }

template<typename dataType>
void DataFrame<dataType>::SwapRowsData(const unsigned& i, const unsigned& j) { data.swap_rows(i, j); }
template<typename dataType>
void DataFrame<dataType>::SwapColsData(const unsigned& i, const unsigned& j) { data.swap_cols(i, j); }

template<typename dataType>
arma::irowvec DataFrame<dataType>::GetTargetMatrixRow(const unsigned& i) const {
    if ( !targetMatrix.size() ) {
        cout << "Usage: Target matrix is not activated!!" << endl << "       You must SetTargetMatrix() function!!" << endl << endl;
        exit(1);
    }
    else
        return targetMatrix.row(i);
}
template<typename dataType>
arma::ivec DataFrame<dataType>::GetTargetMatrixCol(const unsigned& j) const {
    if ( !targetMatrix.size() ) {
        cout << "Usage: Target matrix is not activated!!" << endl << "       You must SetTargetMatrix() function!!" << endl << endl;
        exit(1);
    }
    else
        return targetMatrix.col(j);
}


template<typename dataType>
void DataFrame<dataType>::CopyTarget(const arma::ivec& target) {
    this->target.copy_size(target);
    this->target = target;
}
template<typename dataType>
void DataFrame<dataType>::CopyTargetMatrix(const arma::imat& targetM) {
    targetMatrix.copy_size(targetM);
    targetMatrix = targetM;
}



template<typename dataType>
void DataFrame<dataType>::LinearScalingEachFeatures(DataFrame<double>& x) {
    x.SetN(N);
    x.SetDimension(dimension);
    x.SetIsTarget(isTarget);
    x.SetDataSize(N, dimension);
    x.CopyTarget(target);
    x.CopyTargetMatrix(targetMatrix);

    arma::Row<dataType> _max(dimension);
    arma::Row<dataType> _min(dimension);
    _max = max(data);
    _min = min(data);

    for (unsigned j=0; j<dimension; j++) {
        if ( _max(j) != _min(j) ) {
            for (unsigned i=0; i<N; i++) {
                double temp = ((double) data(i, j) - (double) _min(j)) / ((double) _max(j) - (double) _min(j));
                x.SetData(temp, i, j);
            }
        }
        else {
            for (unsigned i=0; i<N; i++) {
                double temp = 0.0;
                x.SetData(temp, i, j);
            }
        }
    }
}


template<typename dataType>
void DataFrame<dataType>::LinearScalingEachFeatures(DataFrame<dataType>& v, DataFrame<dataType>& t,
            DataFrame<double>& x, DataFrame<double>& y, DataFrame<double>& z) {

    x.SetN(N);                          y.SetN(v.GetN());                           z.SetN(t.GetN());
    x.SetDimension(dimension);          y.SetDimension(dimension);                  z.SetDimension(dimension);
    x.SetIsTarget(isTarget);            y.SetIsTarget(v.IsTarget());                z.SetIsTarget(t.IsTarget());
    x.SetDataSize(N, dimension);        y.SetDataSize(v.GetN(), dimension);         z.SetDataSize(t.GetN(), dimension);
    x.CopyTarget(target);               y.CopyTarget(v.GetTarget());                z.CopyTarget(t.GetTarget());
    x.CopyTargetMatrix(targetMatrix);   y.CopyTargetMatrix(v.GetTargetMatrix());    z.CopyTargetMatrix(t.GetTargetMatrix());

    arma::Row<dataType> _max(dimension), _min(dimension);
    _max = max(data);       _min = min(data);

    for (unsigned j=0; j<dimension; j++) {
        if ( _max(j) != _min(j) ) {
            for (unsigned i=0; i<N; i++) {
                double temp = ((double) data(i, j) - (double) _min(j)) / ((double) _max(j) - (double) _min(j));
                x.SetData(temp, i, j);
            }

            for (unsigned i=0; i<v.GetN(); i++) {
                double temp = ((double) v.GetData(i, j) - (double) _min(j)) / ((double) _max(j) - (double) _min(j));
                y.SetData(temp, i, j);
            }

            for (unsigned i=0; i<t.GetN(); i++) {
                double temp = ((double) t.GetData(i, j) - (double) _min(j)) / ((double) _max(j) - (double) _min(j));
                z.SetData(temp, i, j);
            }
        }
        else {
            double temp = 0.0;
            for (unsigned i=0; i<N; i++) x.SetData(temp, i, j);
            for (unsigned i=0; i<v.GetN(); i++) y.SetData(temp, i, j);
            for (unsigned i=0; i<t.GetN(); i++) z.SetData(temp, i, j);
        }
    }
}



template<typename dataType>
void DataFrame<dataType>::NormalizationEachFeatures(DataFrame<double>& x) {
    x.SetN(N);
    x.SetDimension(dimension);
    x.SetIsTarget(isTarget);
    x.SetDataSize(N, dimension);
    x.CopyTarget(target);
    x.CopyTargetMatrix(targetMatrix);

    arma::vec mean(dimension), mean2(dimension);
    mean.zeros();
    mean2.zeros();
    for (unsigned j=0; j<dimension; j++) {
        for (unsigned i=0; i<N; i++) {
            mean(j) += (double) data(i, j);
            mean2(j) += (double) data(i, j) * (double) data(i, j);
        }
    }
    mean /= (double) N;
    mean2 /= (double) N;
    arma::vec stdev = sqrt(mean2 - square(mean));


    for (unsigned j=0; j<dimension; j++) {
        if ( stdev(j) != 0.0 ) {
            for (unsigned i=0; i<N; i++) {
                double temp = ((double) data(i, j) - mean(j)) / stdev(j);
                x.SetData(temp, i, j);
            }
        }
        else {
            for (unsigned i=0; i<N; i++) {
                double temp = 0.0;
                x.SetData(temp, i, j);
            }
        }
    }
}

template<typename dataType>
void DataFrame<dataType>::NormalizationEachFeatures(DataFrame<dataType>& v, DataFrame<dataType>& t,
            DataFrame<double>& x, DataFrame<double>& y, DataFrame<double>& z) {

    x.SetN(N);                          y.SetN(v.GetN());                           z.SetN(t.GetN());
    x.SetDimension(dimension);          y.SetDimension(dimension);                  z.SetDimension(dimension);
    x.SetIsTarget(isTarget);            y.SetIsTarget(v.IsTarget());                z.SetIsTarget(t.IsTarget());
    x.SetDataSize(N, dimension);        y.SetDataSize(v.GetN(), dimension);         z.SetDataSize(t.GetN(), dimension);
    x.CopyTarget(target);               y.CopyTarget(v.GetTarget());                z.CopyTarget(t.GetTarget());
    x.CopyTargetMatrix(targetMatrix);   y.CopyTargetMatrix(v.GetTargetMatrix());    z.CopyTargetMatrix(t.GetTargetMatrix());

    arma::vec mean(dimension), mean2(dimension);
    mean.zeros();
    mean2.zeros();
    for (unsigned j=0; j<dimension; j++) {
        for (unsigned i=0; i<N; i++) {
            mean(j) += (double) data(i, j);
            mean2(j) += (double) data(i, j) * (double) data(i, j);
        }
    }
    mean /= (double) N;
    mean2 /= (double) N;
    arma::vec stdev = sqrt(mean2 - square(mean));


    for (unsigned j=0; j<dimension; j++) {
        if ( stdev(j) != 0.0 ) {
            for (unsigned i=0; i<N; i++) {
                double temp = ((double) data(i, j) - mean(j)) / stdev(j);
                x.SetData(temp, i, j);
            }

            for (unsigned i=0; i<v.GetN(); i++) {
                double temp = ((double) v.GetData(i, j) - mean(j)) / stdev(j);
                y.SetData(temp, i, j);
            }

            for (unsigned i=0; i<t.GetN(); i++) {
                double temp = ((double) t.GetData(i, j) - mean(j)) / stdev(j);
                z.SetData(temp, i, j);
            }
        }
        else {
            double temp = 0.0;
            for (unsigned i=0; i<N; i++) x.SetData(temp, i, j);
            for (unsigned i=0; i<v.GetN(); i++) y.SetData(temp, i, j);
            for (unsigned i=0; i<t.GetN(); i++) z.SetData(temp, i, j);
        }
    }
}





template<typename dataType>
void DataFrame<dataType>::TransformBinaryData() {
    for (unsigned i=0; i<N; i++)
        for (unsigned j=0; j<dimension; j++)
            if ( data(i, j) != 0 )
                data(i, j) = 1;
}


template<typename dataType>
void DataFrame<dataType>::SplitValidationSet(DataFrame<dataType>& valid, const unsigned& n_valid) {

    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //  Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //  link the Generator to the distribution

    arma::vec rand_data(N);
    for (unsigned n=0; n<N; n++)
        rand_data(n) = urnd();
    arma::uvec shuffleindex = sort_index(rand_data);
    arma::uvec trainindex = shuffleindex.head_rows(N-n_valid);
    arma::uvec validindex = shuffleindex.tail_rows(n_valid);


    valid.SetN(n_valid);
    valid.SetDimension(dimension);
    valid.SetIsTarget(isTarget);
    valid.SetSubset(data, target, targetMatrix, validindex);

    data = data.rows(trainindex);
    target = target(trainindex);
    targetMatrix = targetMatrix.rows(trainindex);
    N -= n_valid;
}

template<typename dataType>
void DataFrame<dataType>::SplitValidTestSet(DataFrame<dataType>& valid, const unsigned& n_valid, DataFrame<dataType>& test, const unsigned& n_test) {

    boost::random::uniform_real_distribution<> uniform_real_dist(0, 1);        //  Choose a distribution
    boost::random::variate_generator<boost::mt19937 &,
        boost::random::uniform_real_distribution<> > urnd(rng, uniform_real_dist);    //  link the Generator to the distribution

    arma::vec rand_data(N);
    for (unsigned n=0; n<N; n++)
        rand_data(n) = urnd();
    arma::uvec shuffleindex = sort_index(rand_data);
    arma::uvec trainindex = shuffleindex.head_rows(N-n_valid);
    arma::uvec testindex = shuffleindex.tail_rows(n_test);

    arma::uvec valid_temp(n_valid);
    for (unsigned n=0; n<n_valid; n++)
        valid_temp(n) = N-n_valid-n_test + n;
    arma::uvec validindex = shuffleindex(valid_temp);

    valid.SetN(n_valid);
    valid.SetDimension(dimension);
    valid.SetIsTarget(isTarget);
    valid.SetSubset(data, target, targetMatrix, validindex);

    test.SetN(n_test);
    test.SetDimension(dimension);
    test.SetIsTarget(isTarget);
    test.SetSubset(data, target, targetMatrix, testindex);

    data = data.rows(trainindex);
    target = target(trainindex);
    targetMatrix = targetMatrix.rows(trainindex);
    N -= n_valid + n_test;
}







}

#endif
