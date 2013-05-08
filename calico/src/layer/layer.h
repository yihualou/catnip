#ifndef LAYER_H
#define LAYER_H

#include "global_defines.h"

#include <viennacl/matrix.hpp>

#include "activation/activation.h"

namespace catnip {
  class Layer {
    public:
      Layer(Activator& activator, unsigned int k);
      Layer(const Layer& other);
      virtual ~Layer() {};

      Layer& operator=(const Layer& other);
      virtual Layer* clone() const;

      unsigned int get_k();
      const Activator& get_activator();
      viennacl::vector<float>& get_value();

      virtual void activate();
      virtual void activate(viennacl::vector<float>& derivatives);

    protected:
      unsigned int k_;
      Activator& activator_;
      viennacl::vector<float> value_;
  };

  class BiasedLayer : public Layer {
    public:
      BiasedLayer(Activator& activator, unsigned int k);
      BiasedLayer(const BiasedLayer& other);

      BiasedLayer& operator=(const BiasedLayer& other);
      virtual BiasedLayer* clone() const;

      viennacl::vector<float>& get_biases();

      void activate();
      void activate(viennacl::vector<float>& derivatives);

    protected:
      viennacl::vector<float> biases_;
  };

  std::ostream& operator<<(std::ostream & stream, Layer& layer);
}

#endif