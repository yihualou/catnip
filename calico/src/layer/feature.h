#ifndef FEATURE_MAP_H
#define FEATURE_MAP_H

#include "global_defines.h"

#include <boost/scoped_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <viennacl/matrix.hpp>
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/range.hpp"

#include "activation/activation.h"
#include "layer/layer.h"

namespace catnip {
  class FeatureMapArray {
    public:
      typedef viennacl::matrix_range<viennacl::matrix<float> > MatrixRange;

      FeatureMapArray(viennacl::vector<float>& wrapped, unsigned int width, unsigned int height, unsigned int num_feature_maps);
      FeatureMapArray(const FeatureMapArray& other);
      virtual ~FeatureMapArray() {};

      boost::ptr_vector<MatrixRange>& get_feature_maps();
      MatrixRange& operator[](const int i);

    protected:
      unsigned int width_;
      unsigned int height_;
      unsigned int num_feature_maps_;

      viennacl::vector<float>& wrapped_;

      viennacl::matrix<float> matrix_alias_;
      boost::ptr_vector<MatrixRange> feature_maps_;

    private:
      void init_feature_maps();
  };

  class FeatureMapLayer : public Layer {
    public:
      FeatureMapLayer(Activator& activator, unsigned int width, unsigned int height, unsigned int num_feature_maps);
      FeatureMapLayer(const FeatureMapLayer& other);
      virtual ~FeatureMapLayer() {};

      FeatureMapLayer* clone() const;

      unsigned int get_width() { return width_; }
      unsigned int get_height() { return height_; }
      unsigned int get_num_feature_maps() { return num_feature_maps_; }
      FeatureMapArray::MatrixRange& get_feature_map(const int i);
      viennacl::matrix<float>& get_bias(const int i);

      virtual void activate();
      virtual void activate(viennacl::vector<float>& derivatives);

    protected:
      unsigned int width_;
      unsigned int height_;
      unsigned int num_feature_maps_;
      boost::ptr_vector<viennacl::matrix<float> > biases_;

    private:
      FeatureMapArray arr_;

      void init_feature_maps();
  };
}

#endif