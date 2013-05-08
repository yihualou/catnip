#ifndef RBM_H
#define RBM_H

#include "global_defines.h"

#include <boost/scoped_ptr.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <string>
#include <viennacl/matrix.hpp>

#include "activation/activation.h"
#include "connection/connection.h"
#include "layer/layer.h"
#include "network/network.h"
#include "util/ocl.h"

namespace catnip {
  struct RbmProgram : public OclProgram<RbmProgram> {
    static inline std::string program_name() {
      return "catnip_rbm";
    }

    static inline std::string relative_path() {
      return "/rbm/rbm.cl";
    }

    static inline std::string* kernel_names(unsigned int& num_kernels) {
      num_kernels = 1;
      static std::string kernel_names[1] = { "rbm_activate"};
      return kernel_names;
    }
  };

  struct RbmActivator : Activator {
    void activate(const viennacl::vector<float>& input, viennacl::vector<float>& output);
    void derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes);
  };

  class RbmNetwork : public Network {
    public:
      RbmNetwork(unsigned int ki, unsigned int ko);

      BiasedLayer& get_input_layer();
      BiasedLayer& get_output_layer();

      FullConnection& get_connection();

      void activate(const viennacl::vector<float>& input);
    
    private:
      RbmActivator activator_;
      boost::scoped_ptr<BiasedLayer> visible_layer_;
      boost::scoped_ptr<BiasedLayer> hidden_layer_;
      boost::scoped_ptr<FullConnection> connection_;
  };
}

#endif