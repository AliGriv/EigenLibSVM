
# EigenLibSVM
Andrej Karpathy
1 May 2012

This is a small C++ wrapper to call libsvm if you use the Eigen matrix library.
Dependencies consist of libsvm and eigen3 library.

## Usage

vector<int> yhat;
SVMClassifier svm;
svm.train(X, y);
svm.test(X, yhat);

where X is an Eigen::MatrixXf NxD matrix, y is an Eigen::MatrixXf Nx1 matrix of
labels (-1 or 1), or a vector<int> of labels.
See more for included demo.

## Install

$ sudo apt-get install libsvm-dev
$ sudo apt-get install libeigen3-dev
$ git clone <this project's .git>
$ cd eigenlibsvm/build
$ cmake ..
$ make
$ ./
$ ./svm_test

where the last line will run a tiny demo that makes sure everything installed ok
(it runs almost instantly)

## License
BSD