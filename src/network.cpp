// Copyright 2018 ishan@khatri.io

#include "network.h"

namespace network{

using Eigen::Matrix;
using Eigen::Dynamic;
using std::random_shuffle;
using std::cout;
using std::endl;
using std::get;
using std::make_tuple;

// Network class functions

Network::Network(vector<int> sizes){
  num_layers = sizes.size();
  layer_sizes = sizes;
  
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0,1.0);
  
  for(int layer=1; layer<num_layers; layer++){
    VectorXf layer_biases(layer_sizes[layer]);
    for(int neuron=0; neuron<layer_sizes[layer]; neuron++){
      layer_biases(neuron) = distribution(generator);
    }
    biases.push_back(layer_biases);
  }

  for(int layer=0; layer<num_layers-1; layer++){
    MatrixXf layer_weights(layer_sizes[layer+1], layer_sizes[layer]);
    for(int neuron_l1=0; neuron_l1<layer_sizes[layer+1]; neuron_l1++){
      for(int neuron_l0=0; neuron_l0<layer_sizes[layer]; neuron_l0++){
        layer_weights(neuron_l1, neuron_l0) = distribution(generator);
      }
    }
    weights.push_back(layer_weights);
  }
}

VectorXf Network::FeedForward(VectorXf a) {
  MatrixXf r;
  for (int i = 0; i < num_layers; i++) {
    r = sigmoid((weights[i] * a) + biases[i]);
  }
  return r;
}

void Network::SGD(vector<data> training_data, int epochs, int mini_batch_size, float learning_rate, vector<data> test_data) {
  int n_test = test_data.size();
  int n = training_data.size();

  for(int i=0; i<epochs; i++){
    // Suffle the training data
    random_shuffle(training_data.begin(), training_data.end());
    
    // Create mini_batches
    vector<vector<data>> mini_batches;
    auto k = training_data.begin();
    while(k != training_data.end()){
      vector<data> mini_batch;
      for(int i=0; i<mini_batch_size; i++){
        mini_batch.push_back(*(k++));
      }
      mini_batches.push_back(mini_batch);
    }
    
    // Update each mini_batch
    for(auto mini_batch : mini_batches){
      update_mini_batch(mini_batch, learning_rate);
    }

    // Evaluate against test data if provided
    if(n_test != 0){
      cout << "Epoch " << i << ": " << evaluate(test_data) << " / " << n_test << endl;
    }
    else{
      cout << "Epoch " << i << " complete" << endl;
    }
  }
}

void Network::update_mini_batch(vector<data> mini_batch, float learning_rate){
  vector<VectorXf> nabla_b;
  for(int layer=1; layer<num_layers; layer++){
    VectorXf per_layer_nabla_b = VectorXf::Zero(layer_sizes[layer]);
    nabla_b.push_back(per_layer_nabla_b);
  }
  vector<MatrixXf> nabla_w;
  for(int layer=0; layer<num_layers-1; layer++){
    MatrixXf per_layer_nabla_w = MatrixXf::Zero(layer_sizes[layer+1], layer_sizes[layer]);
    nabla_w.push_back(per_layer_nabla_w);
  }

  for(auto d : mini_batch){
    auto b = backprop(d);
    vector<VectorXf> delta_nabla_b = get<0>(b);
    vector<MatrixXf> delta_nabla_w = get<1>(b);

    auto it1 = nabla_b.begin();
    auto it2 = delta_nabla_b.begin();
    while(it1 != nabla_b.end() && it2 != delta_nabla_b.end()){
      *it1 = *it1 + *it2;
      it1++;
      it2++;
    }

    auto it3 = nabla_w.begin();
    auto it4 = delta_nabla_w.begin();
    while(it3 != nabla_w.end() && it4 != delta_nabla_w.end()){
      *it3 = *it3 + *it4;
      it3++;
      it4++;
    }
  }

  for(int i=0; i<biases.size(); i++){
    biases[i] = biases[i] - (learning_rate / mini_batch.size())*nabla_b[i];
  }

  for(int i=0; i<weights.size(); i++){
    weights[i] = weights[i] - (learning_rate / mini_batch.size())*nabla_w[i];
  }
}

tuple<vector<VectorXf>, vector<MatrixXf>> Network::backprop(data d){
  vector<VectorXf> nabla_b;
  for(int layer=1; layer<num_layers; layer++){
    VectorXf per_layer_nabla_b = VectorXf::Zero(layer_sizes[layer]);
    nabla_b.push_back(per_layer_nabla_b);
  }
  vector<MatrixXf> nabla_w;
  for(int layer=0; layer<num_layers-1; layer++){
    MatrixXf per_layer_nabla_w = MatrixXf::Zero(layer_sizes[layer+1], layer_sizes[layer]);
    nabla_w.push_back(per_layer_nabla_w);
  }

  // Feed forward
  MatrixXf activation = get<0>(d);
  vector<MatrixXf> activations = {activation};
  vector<MatrixXf> zs;

  for(int i=0; i<biases.size(); i++){
    MatrixXf z = weights[i]*activation + biases[i];
    zs.push_back(z);
    activation = sigmoid(z);
    activations.push_back(activation);
  }

  // Backward pass
  VectorXf delta = cost_derivative(*activations.end(), get<1>(d)) * sigmoid_prime(*zs.end());
  *nabla_b.end() = delta;
  *nabla_w.end() = delta * (*(activations.end()-1)).transpose();

  // for(int l=num_layers-2; l>1; l--){
  //   MatrixXf z = zs[l];
  //   MatrixXf sp = sigmoid_prime(z);
  //   delta = weights[l+1].transpose() * delta * sp;
  // }
}

int Network::evaluate(vector<data> test_data){
  return 0;
}

VectorXf Network::cost_derivative(VectorXf output_activations, int y){
  return (output_activations-convert_output(y));
}

VectorXf Network::convert_output(int output){
  VectorXf activation = VectorXf::Zero(*layer_sizes.end());
  activation[output] = 1;
  return activation;
}

// Misc functions

MatrixXf sigmoid(MatrixXf z){
  // using a faster approximation of the sigmoid function
  // f(x) = x / (1 + abs(x))
  MatrixXf r = z;
  for(int i=0; i< z.size(); i++){
    float x = r.data()[i];
    r.data()[i] = x / (1 + fabs(x));
  }
  return r;
}

MatrixXf sigmoid_prime(MatrixXf z){
  MatrixXf one = MatrixXf::Constant(z.rows(), z.cols(), 1);
  return sigmoid(z) * (one-sigmoid(z));
}

} // namespace network

int main(){
  using network::Network;
  using std::cout;
  using std::vector;
  using std::endl;
  vector<int> s = {4, 3, 2};
  Network n = Network(s);
  cout<<"Baises for layers of size 3 and 2"<<endl;
  for(auto x : n.biases)
    cout<<x<<endl;

  cout<<"Weight matricies"<<endl;
  for(auto x : n.weights)
    cout<<x<<endl;
  return 0;
}
