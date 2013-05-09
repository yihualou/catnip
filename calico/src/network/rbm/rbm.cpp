#include "global_defines.h"

#include <boost/unordered_set.hpp>
#include <viennacl/ocl/backend.hpp>

#include "rbm.h"

namespace catnip {
  std::map<cl_context, bool> RbmProgram::init_done;

  void RbmActivator::activate(const viennacl::vector<float>& input, viennacl::vector<float>& output) {
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(RbmProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("rbm_activate");
    viennacl::ocl::enqueue(k(input, output, cl_uint(input.size())));
  }

  void RbmActivator::derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes) {
  }

  RbmNetwork::RbmNetwork(unsigned int ki, unsigned int ko) {
    RbmProgram::init();

    visible_layer_.reset(new BiasedLayer(activator_, ki));
    hidden_layer_.reset(new BiasedLayer(activator_, ko)); 
    connection_.reset(new FullConnection(ki, ko));
  } 

  BiasedLayer& RbmNetwork::get_input_layer() {
    return *visible_layer_;
  }
  
  BiasedLayer& RbmNetwork::get_output_layer() {
    return *hidden_layer_;
  }

  FullConnection& RbmNetwork::get_connection() {
    return *connection_;
  }

  void RbmNetwork::activate(const viennacl::vector<float>& input) {
    connection_->layer_propogate(*visible_layer_, *hidden_layer_);
    hidden_layer_->activate();
  }
}