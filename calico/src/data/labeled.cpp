#include "labeled.h"

namespace catnip {
  LabeledDataset::LabeledDataset(int input_size, int output_size)
    : input_size_(input_size), output_size_(output_size) {}

  int LabeledDataset::input_size() const {
    return input_size_;
  }

  int LabeledDataset::output_size() const {
    return output_size_;
  }

  int LabeledDataset::size() const {
    return input_store_.size();
  }

  const viennacl::vector<float>& LabeledDataset::get_input(int i) const {
    return input_store_[i];
  }

  const viennacl::vector<float>&  LabeledDataset::get_output(int i) const {
    return output_store_[i];
  }

  void LabeledDataset::add_observation(const viennacl::vector<float>& in, const viennacl::vector<float>& out) {
    viennacl::vector<float>* vin = new viennacl::vector<float>(in);
    viennacl::vector<float>* vout = new viennacl::vector<float>(out);

    input_store_.push_back(vin);
    output_store_.push_back(vout);
  }

  void LabeledDataset::add_observation(const std::vector<float>& in, const std::vector<float>& out) {
    viennacl::vector<float>* vin = new viennacl::vector<float>(in.size());
    viennacl::vector<float>* vout = new viennacl::vector<float>(out.size());

    fast_copy(in, *vin);
    fast_copy(out, *vout);

    input_store_.push_back(vin);
    output_store_.push_back(vout);
  }
}