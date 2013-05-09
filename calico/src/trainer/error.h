#ifndef ERROR_H_
#define ERROR_H_

#include <viennacl/vector.hpp>

#include "util/ocl.h"

namespace catnip {
  struct ErrorProgram : public OclProgram<ErrorProgram> {
    static std::map<cl_context, bool> init_done;

    static inline std::string program_name() {
      return "catnip_error";
    }

    static inline std::string relative_path() {
      return "/trainer/error.cl";
    }

    static inline std::string* kernel_names(unsigned int& num_kernels) {
      num_kernels = 1;
      static std::string kernel_names[1] = { "rms" };
      return kernel_names;
    }
  };

  struct ErrorFunction {
      virtual void evaluate(const viennacl::vector<float>& predicted, const viennacl::vector<float>& actual, viennacl::vector<float>& error) = 0;
  };

  struct RmsErrorFunction : public ErrorFunction {
      void evaluate(const viennacl::vector<float>& predicted, const viennacl::vector<float>& actual, viennacl::vector<float>& error);
  };

  struct LinearErrorFunction : public ErrorFunction {
      void evaluate(const viennacl::vector<float>& predicted, const viennacl::vector<float>& actual, viennacl::vector<float>& error);
  };
}

#endif // ERROR_H_