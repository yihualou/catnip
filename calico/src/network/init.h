#ifndef INIT_H_
#define INIT_H_

#include "global_defines.h"

#include <string>
#include <viennacl/matrix.hpp>
#include <viennacl/compressed_matrix.hpp>

#include "connection/convolution.h"
#include "connection/connection.h"
#include "layer/feature.h"
#include "layer/layer.h"
#include "util/ocl.h"

namespace catnip {
  struct InitializerProgram : public OclProgram<InitializerProgram> {
    static inline std::string program_name() {
      return "catnip_init";
    }

    static inline std::string relative_path() {
      return "/network/init.cl";
    }

    static inline std::string* kernel_names(unsigned int& num_kernels) {
      num_kernels = 2;
      static std::string kernel_names[2] = { "random_initialize", "fan_in_initialize" };
      return kernel_names;
    }
  };

  struct Initializer {
    virtual void initialize(Layer& layer) const = 0;
    virtual void initialize(Connection& connection) const = 0;
  };

  class RandomInitializer : public Initializer {
    public:
      RandomInitializer(float magnitude);

      virtual void initialize(Layer& layer) const;
      virtual void initialize(Connection& connection) const;

    private:
      float magnitude_;
  };

  class FanInInitializer : public Initializer {
    public:
      FanInInitializer(float magnitude);

      virtual void initialize(Layer& layer) const;
      virtual void initialize(Connection& connection) const;

    private:
      void initialize(BiasedLayer& layer) const;
      void initialize(FeatureMapLayer& layer) const;

      void initialize(FullConnection& connection) const;
      void initialize(IdentityConnection& connection) const;
      void initialize(SparseConnection& connection) const;
      void initialize(ConvolutionConnection& connection) const;
      void initialize(MaxPoolingConnection& connection) const;

      float magnitude_;
  };
}

#endif // INIT_H_