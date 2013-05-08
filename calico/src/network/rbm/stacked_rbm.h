#ifndef STACKED_RBM_H_
#define STACKED_RBM_H_

#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>

#include "connection/connection.h"
#include "layer/layer.h"
#include "network/feedforward/feedforward.h"

namespace catnip {
  class StackedRbmNetwork : public FeedForwardNetwork {
    public:
      StackedRbmNetwork(std::vector<unsigned int> dimensions);
  };
}

#endif // STACKED_RBM_H_