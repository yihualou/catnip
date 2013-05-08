#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "global_defines.h"

#include <string>
#include <boost/unordered_set.hpp>
#include <viennacl/vector.hpp>

#include "util/ocl.h"

namespace catnip {
  struct ActivationProgram : public OclProgram<ActivationProgram> {
    static inline std::string program_name() {
      return "catnip_activation";
    }

    static inline std::string relative_path() {
      return "/activation/activation.cl";
    }

    static inline std::string* kernel_names(unsigned int& num_kernels) {
      num_kernels = 8;
      static std::string kernel_names[8] = {
        "linear_activate",
        "linear_derivatives",
        "tanh_activate", 
        "tanh_derivatives", 
        "sigmoid_activate", 
        "sigmoid_derivatives", 
        "softmax_activate",  
        "softmax_derivatives"
      };
      return kernel_names;
    }
  };

  struct Activator {
    virtual void activate(const viennacl::vector<float>& input, viennacl::vector<float>& output) = 0;
    virtual void derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes) = 0;
  };

  struct LinearActivator : Activator {
    LinearActivator(float constant);

    void activate(const viennacl::vector<float>& input, viennacl::vector<float>& output);
    void derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes);

  private:
    float constant_;
  };

  struct SigmoidActivator : Activator {
    SigmoidActivator();

    void activate(const viennacl::vector<float>& input, viennacl::vector<float>& output);
    void derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes);
  };

  struct TanhActivator: Activator {
    TanhActivator();

    void activate(const viennacl::vector<float>& input, viennacl::vector<float>& output);
    void derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes);
  };

  struct SoftmaxActivator: Activator {
    SoftmaxActivator();

    void activate(const viennacl::vector<float>& input, viennacl::vector<float>& output);
    void derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes);
  };
}

#endif