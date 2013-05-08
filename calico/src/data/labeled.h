#ifndef LABELED_H_
#define LABELED_H_

#include "global_defines.h"

#include <string>
#include <boost/array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/ocl/backend.hpp>

namespace catnip {
  class LabeledDataset {
    public:
      LabeledDataset(int input_size, int output_size);
      virtual ~LabeledDataset() {}

      int input_size() const;
      int output_size() const;
      int size() const;

      const viennacl::vector<float>& get_input(int i) const;
      const viennacl::vector<float>& get_output(int i) const;

      void add_observation(const viennacl::vector<float>& in, const viennacl::vector<float>& out);

      template<int ki, int ko>
      void add_observation(const boost::array<float, ki>& in, const boost::array<float, ko>& out) {
        viennacl::vector<float>* vin = new viennacl::vector<float>(ki);
        viennacl::vector<float>* vout = new viennacl::vector<float>(ko);

        fast_copy(in, *vin);
        fast_copy(out, *vout);

        input_store_.push_back(vin);
        output_store_.push_back(vout);
      }

      void add_observation(const std::vector<float>& in, const std::vector<float>& out);
      
    protected:
      int input_size_;
      int output_size_;
      boost::ptr_vector<viennacl::vector<float> > input_store_; 
      boost::ptr_vector<viennacl::vector<float> > output_store_; 
  };
}

#endif // LABELED_H_