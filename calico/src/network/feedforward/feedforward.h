#ifndef FEEDFORWARD_NETWORK_H
#define FEEDFORWARD_NETWORK_H

#include "global_defines.h"

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "layer/layer.h"
#include "connection/connection.h"
#include "network/network.h"

namespace catnip {
  class FeedForwardNetwork;

  template<>
  struct IterableNetworkTraits<FeedForwardNetwork> {
    typedef boost::ptr_vector<Layer>::iterator LayerIterator;
    typedef boost::ptr_vector<Layer>::reverse_iterator LayerReverseIterator;
    typedef boost::ptr_vector<Connection>::iterator ConnectionIterator;
    typedef boost::ptr_vector<Connection>::reverse_iterator ConnectionReverseIterator;
  };

  class FeedForwardNetwork : public IterableNetwork<FeedForwardNetwork> {
    public:
      FeedForwardNetwork::FeedForwardNetwork();
      FeedForwardNetwork(const FeedForwardNetwork& other);
      FeedForwardNetwork& operator=(const FeedForwardNetwork& other);

      int get_num_layers() { return layers_.size(); }
      int get_num_connections() { return connections_.size(); }

      Layer& get_layer(const int i);
      Connection& get_connection(const int i);

      Layer& get_input_layer();
      Layer& get_output_layer();

      LayerIterator layers_begin();
      LayerIterator layers_end();
      LayerReverseIterator layers_rbegin();
      LayerReverseIterator layers_rend();

      ConnectionIterator connections_begin();
      ConnectionIterator connections_end();
      ConnectionReverseIterator connections_rbegin();
      ConnectionReverseIterator connections_rend();

      void initialize(const Initializer& initializer);
      void activate(const viennacl::vector<float>& input);

      void set_input_layer(Layer* layer);
      void add_layer(Connection* connection, Layer* layer);
      void remove_layer(LayerIterator iterator);

    protected:
      boost::ptr_vector<Layer> layers_;
      boost::ptr_vector<Connection> connections_;
  };
}

#endif