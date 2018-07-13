// Copyright 2018 ishan@khatri.io

#include "network.h"

namespace network{

using Eigen::Matrix;
using Eigen::Dynamic;

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

// Misc functions

// using a faster approximation of the sigmoid function
// f(x) = x / (1 + abs(x))
MatrixXf sigmoid(MatrixXf z){
  MatrixXf r = z;
  for(int i=0; i< z.size(); i++){
    float x = r.data()[i];
    r.data()[i] = x / (1 + fabs(x));
  }
  return r;
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
