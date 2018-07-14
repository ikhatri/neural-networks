// Copyright 2018 ishan@khatri.io

#ifndef NETWORK_H_

#include <iostream>
#include <vector>
#include <random>
#include <tuple>

#ifdef _WIN32
#include <Eigen/dense>
#endif

#ifdef linux
#include <eigen3/Eigen/Dense>
#endif

namespace network{

using std::vector;
using std::tuple;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class Network{
  public:
    int num_layers;
    vector<int> layer_sizes;
    vector<VectorXf> biases;
    vector<MatrixXf> weights;
    Network(vector<int> s);
    VectorXf FeedForward(VectorXf a);
    void SGD(tuple<int, int> test_data = nullptr);
};

} // namespace network

#endif
