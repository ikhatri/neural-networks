// Copyright 2018 ishan@khatri.io

#ifndef NETWORK_H_

#include "MNISTParser.h"

#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#include <Eigen/dense>
#endif

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#endif

namespace network{

using std::vector;
using std::tuple;
using Eigen::MatrixXf;
using Eigen::VectorXf;

typedef tuple<VectorXf, int> data;

class Network{
  public:
    int num_layers;
    vector<int> layer_sizes;
    vector<VectorXf> biases;
    vector<MatrixXf> weights;
    Network(vector<int> s);
    VectorXf FeedForward(VectorXf a);
    void SGD(vector<data> training_data, int epochs, int mini_batch_size, float learning_rate, vector<data> test_data = vector<data>());
    void update_mini_batch(vector<data> mini_batch, float learning_rate);
    int evaluate(vector<data> test_data);
    tuple<vector<VectorXf>, vector<MatrixXf>> backprop(data d);
    VectorXf cost_derivative(VectorXf output_activations, int y);
    VectorXf convert_output(int output);
    int max_index(VectorXf a);
};
// Misc functions
MatrixXf sigmoid(MatrixXf z);
MatrixXf sigmoid_prime(MatrixXf z);

} // namespace network

#endif
