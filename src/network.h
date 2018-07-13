// Copyright 2018 ishan@khatri.io

#ifndef NETWORK_H_

#include <iostream>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>

namespace network{

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class Network{
  public:
    int num_layers;
    vector<int> layer_sizes;
    vector<VectorXf> biases;
    vector<MatrixXf> weights;
    Network(vector<int> s);
};

} // namespace network

#endif
