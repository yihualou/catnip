#include "layer.h"

namespace catnip {
  Layer::Layer(Activator& activator, unsigned int k): activator_(activator), k_(k) {
    value_ = viennacl::zero_vector<float>(k);
  }

  Layer::Layer(const Layer& other) : activator_(other.activator_), k_(other.k_) {
   value_ = viennacl::zero_vector<float>(k_);
  }

  Layer& Layer::operator=(const Layer& other) {
    activator_ = other.activator_;
    k_ = other.k_;
    value_ = other.value_;
    return *this;
  }

  Layer* Layer::clone() const {
    return new Layer(*this);
  }

  unsigned int Layer::get_k() {
    return k_;
  }

  const Activator& Layer::get_activator() {
    return activator_;
  }

  viennacl::vector<float>& Layer::get_value() {
    return value_;
  }

  void Layer::activate() {
    activator_.activate(value_, value_);
  }

  void Layer::activate(viennacl::vector<float>& derivatives) {
    activator_.derivatives(value_, derivatives);
    activator_.activate(value_, value_);
  }

  BiasedLayer::BiasedLayer(Activator& activator, unsigned int k) : Layer(activator, k) {
    biases_ = viennacl::zero_vector<float>(k_);
  }

  BiasedLayer::BiasedLayer(const BiasedLayer& other) : Layer(other) {
    biases_ = other.biases_;
  }

  BiasedLayer& BiasedLayer::operator=(const BiasedLayer& other) {
    activator_ = other.activator_;
    k_ = other.k_;
    value_ = other.value_;
    biases_ = other.biases_;
    return *this;
  }

  BiasedLayer* BiasedLayer::clone() const {
    return new BiasedLayer(*this);
  }

  viennacl::vector<float>& BiasedLayer::get_biases() {
    return biases_;
  }

  void BiasedLayer::activate() {
    value_ += biases_;
    activator_.activate(value_, value_);
  }

  void BiasedLayer::activate(viennacl::vector<float>& derivatives) {
    value_ += biases_;
    activator_.derivatives(value_, derivatives);
    activator_.activate(value_, value_);
  }

  std::ostream& operator<<(std::ostream & stream, Layer& layer) {
    stream << "L" << layer.get_value();
    return stream;
  }
}
