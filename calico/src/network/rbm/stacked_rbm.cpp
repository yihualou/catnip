#include "global_defines.h"

#include <exception>
#include <boost/unordered_set.hpp>
#include <viennacl/ocl/backend.hpp>

#include "rbm.h"
#include "stacked_rbm.h"

namespace catnip {
  StackedRbmNetwork::StackedRbmNetwork(std::vector<unsigned int> dimensions) {
    if (dimensions.size() < 2) {
      throw std::invalid_argument("Dimension vector must contain at least two elements.");
    }
      
    RbmProgram::init();

    RbmActivator activator;
    for (unsigned int i = 0; i < dimensions.size() - 1; ++i) {
      layers_.push_back(new BiasedLayer(activator, dimensions[i]));
      connections_.push_back(new FullConnection(dimensions[i], dimensions[i + 1]));
    }
    layers_.push_back(new BiasedLayer(activator, *dimensions.rbegin())); 
  }
}