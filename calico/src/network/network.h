#ifndef NETWORK_H_
#define NETWORK_H_

#include <boost/scoped_ptr.hpp>
#include "network/init.h"
#include "layer/layer.h"
#include "connection/connection.h"

namespace catnip {
  class Network {
    public:
      virtual Layer& get_input_layer() = 0;
      virtual Layer& get_output_layer() = 0;
      
      virtual void initialize(const Initializer& initializer) = 0;
      virtual void activate(const viennacl::vector<float>& input) = 0;
  };

  template<class Derived>
  struct IterableNetworkTraits;

  template<class Derived>
  class IterableNetwork : public Network {
    public:
      typedef typename IterableNetworkTraits<Derived>::LayerIterator LayerIterator;
      typedef typename IterableNetworkTraits<Derived>::LayerReverseIterator LayerReverseIterator;
      typedef typename IterableNetworkTraits<Derived>::ConnectionIterator ConnectionIterator;
      typedef typename IterableNetworkTraits<Derived>::ConnectionReverseIterator ConnectionReverseIterator;

      virtual Layer& get_input_layer() = 0;
      virtual Layer& get_output_layer() = 0;

      virtual LayerIterator layers_begin() = 0;
      virtual LayerIterator layers_end() = 0;
      virtual LayerReverseIterator layers_rbegin() = 0;
      virtual LayerReverseIterator layers_rend() = 0;

      virtual ConnectionIterator connections_begin() = 0;
      virtual ConnectionIterator connections_end() = 0;
      virtual ConnectionReverseIterator connections_rbegin() = 0;
      virtual ConnectionReverseIterator connections_rend() = 0;

      virtual void initialize(const Initializer& initializer) = 0;
      virtual void activate(const viennacl::vector<float>& input) = 0;
  };
}
#endif // NETWORK_H_