#include "feature.h"

namespace catnip {
  FeatureMapArray::FeatureMapArray(
      viennacl::vector<float>& wrapped,
      unsigned int width, 
      unsigned int height, 
      unsigned int num_feature_maps)
      : wrapped_(wrapped), width_(width), height_(height), num_feature_maps_(num_feature_maps),
        matrix_alias_(wrapped_.handle().opencl_handle().get(), height_ * num_feature_maps_, width_) {

    init_feature_maps();
  }

  FeatureMapArray::FeatureMapArray(const FeatureMapArray& other)
      : wrapped_(other.wrapped_), width_(other.width_), height_(other.height_), num_feature_maps_(other.num_feature_maps_),
        matrix_alias_(wrapped_.handle().opencl_handle().get(), height_ * num_feature_maps_, width_) {
    
    init_feature_maps();
  }

  FeatureMapLayer* FeatureMapLayer::clone() const {
    return new FeatureMapLayer(*this);
  };

  void FeatureMapArray::init_feature_maps() {
    viennacl::range cr(0, width_);
    for (unsigned int i = 0; i < num_feature_maps_; ++i) {
      viennacl::range rr(i * height_, (i + 1) * height_);
      MatrixRange* mr = new MatrixRange(matrix_alias_, rr, cr);
      feature_maps_.push_back(mr);
    }
  }

  boost::ptr_vector<FeatureMapArray::MatrixRange>& FeatureMapArray::get_feature_maps() {
    return feature_maps_; 
  }

  FeatureMapArray::MatrixRange& FeatureMapArray::operator[](const int i) {
    return feature_maps_[i];
  }

  FeatureMapLayer::FeatureMapLayer(Activator& activator, unsigned int width, unsigned int height, unsigned int num_feature_maps) 
      : Layer(activator, width * height * num_feature_maps), 
        width_(width), height_(height), 
        num_feature_maps_(num_feature_maps),
        arr_(value_, width_, height_, num_feature_maps_) { 

    // TODO(ylou): Handle fallback cases when OpenCL isn't enabled?
    for (unsigned int i = 0; i < num_feature_maps; ++i) {
      viennacl::matrix<float>* bias = new viennacl::matrix<float>(height_, width_);
      *bias = viennacl::zero_matrix<float>(height_, width_);
      biases_.push_back(bias);
    }
  }

  FeatureMapLayer::FeatureMapLayer(const FeatureMapLayer& other)
      : Layer(other.activator_, other.k_), 
        width_(other.width_), height_(other.height_), 
        num_feature_maps_(other.num_feature_maps_),
        biases_(other.biases_),
        arr_(value_, width_, height_, num_feature_maps_) {}

  FeatureMapArray::MatrixRange& FeatureMapLayer::get_feature_map(const int i) {
    return arr_[i];
  }

  viennacl::matrix<float>& FeatureMapLayer::get_bias(const int i) {
    return biases_[i];
  }

  void FeatureMapLayer::activate() {
    typedef boost::ptr_vector<viennacl::matrix<float> >::iterator Iterator;
    
    for (Iterator fit = arr_.get_feature_maps().begin(), bit = biases_.begin();
        fit != arr_.get_feature_maps().end(), bit != biases_.end(); 
        ++fit, ++bit) {
      
      viennacl::matrix<float>& fm = *fit;
      viennacl::matrix<float>& bias = *bit;
      
      fm += bias;
    }

    activator_.activate(value_, value_);
  }

  void FeatureMapLayer::activate(viennacl::vector<float>& derivatives) {
    typedef boost::ptr_vector<viennacl::matrix<float> >::iterator Iterator;
    
    for (Iterator fit = arr_.get_feature_maps().begin(), bit = biases_.begin();
        fit != arr_.get_feature_maps().end(), bit != biases_.end(); 
        ++fit, ++bit) {
      
      viennacl::matrix<float>& fm = *fit;
      viennacl::matrix<float>& bias = *bit;
      
      fm += bias;
    }

    activator_.activate(value_, value_);
  }
}
