#include <exception>
#include "feedforward.h"

namespace catnip {
  FeedForwardNetwork::FeedForwardNetwork() {}

  FeedForwardNetwork::FeedForwardNetwork(const FeedForwardNetwork& other) : layers_(other.layers_), connections_(other.connections_) {}

  FeedForwardNetwork& FeedForwardNetwork::operator=(const FeedForwardNetwork& other) {
    layers_ = other.layers_;
    connections_ = other.connections_;
    return *this;
  }

  Layer& FeedForwardNetwork::get_layer(const int i) {
    return layers_[i];
  }

  Connection& FeedForwardNetwork::get_connection(const int i) {
    return connections_[i];
  }

  Layer& FeedForwardNetwork::get_input_layer() {
    return *layers_.begin();
  }
  
  Layer& FeedForwardNetwork::get_output_layer() {
    return *layers_.rbegin();
  }

  FeedForwardNetwork::LayerIterator FeedForwardNetwork::layers_begin() {
    return layers_.begin();  
  };

  FeedForwardNetwork::LayerIterator FeedForwardNetwork::layers_end() {
    return layers_.end();
  };

  FeedForwardNetwork::LayerReverseIterator FeedForwardNetwork::layers_rbegin() {
    return layers_.rbegin();  
  };

  FeedForwardNetwork::LayerReverseIterator FeedForwardNetwork::layers_rend() {
    return layers_.rend();
  };

  FeedForwardNetwork::ConnectionIterator FeedForwardNetwork::connections_begin() {
    return connections_.begin();  
  };

  FeedForwardNetwork::ConnectionIterator FeedForwardNetwork::connections_end() {
    return connections_.end();
  };

  FeedForwardNetwork::ConnectionReverseIterator FeedForwardNetwork::connections_rbegin() {
    return connections_.rbegin();  
  };

  FeedForwardNetwork::ConnectionReverseIterator FeedForwardNetwork::connections_rend() {
    return connections_.rend();
  };

  void FeedForwardNetwork::initialize(const Initializer& initializer) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      initializer.initialize(layers_[i]);
      viennacl::ocl::get_queue().finish();
    }
    for (size_t i = 0; i < connections_.size(); ++i) {
      initializer.initialize(connections_[i]);
      viennacl::ocl::get_queue().finish();
    }
  }

  void FeedForwardNetwork::activate(const viennacl::vector<float>& input) {
    assert(layers_.size() >= 2);
    assert(layers_.size() - 1 == connections_.size());

    layers_.front().get_value() = input;

    for (unsigned int i = 0; i < connections_.size(); ++i) {
      Layer& layer = layers_[i];
      Layer& next_layer = layers_[i + 1];
      Connection& conn = connections_[i];

      layer.activate();
      conn.layer_propogate(layer, next_layer);
    }
    layers_.back().activate();
  }

  void FeedForwardNetwork::set_input_layer(Layer* layer) {
    if (layers_.empty()) {
      layers_.push_back(layer);
    } else {
      boost::ptr_vector<Layer>::auto_type old = layers_.replace(0, layer);
      old.reset();
    } 
  }

  void FeedForwardNetwork::add_layer(Connection* connection, Layer* layer) {
    assert(!layers_.empty());
     
    int ki = layers_.back().get_k();
    int cki = connection->get_ki();
    int cko = connection->get_ko();
    int ko = layer->get_k();

    assert(ki == cki && cko == ko);

    connections_.push_back(connection);
    layers_.push_back(layer);
  }

  void FeedForwardNetwork::remove_layer(FeedForwardNetwork::LayerIterator iterator) {
    if (layers_.begin() <= iterator && iterator < layers_.end()) {
      int dist = iterator - layers_.begin();
      layers_.erase(iterator);
      
      boost::ptr_vector<Connection>::iterator begin = connections_.begin();
      if (dist == 0) {
        connections_.clear();
      } else if (dist == layers_.size() - 1) {
        connections_.erase(connections_.begin() + dist - 1);
      } else {
        connections_.erase(connections_.begin() + dist - 1);
        connections_.erase(connections_.begin() + dist - 1);
      }
    }
  }
}

