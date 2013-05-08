#ifndef BACKPROP_H_
#define BACKPROP_H_

#include "global_defines.h"

#include "network/network.h"
#include "network/feedforward/feedforward.h"
#include "data/labeled.h"
#include "util/ocl.h"

namespace catnip {
  struct BackpropProgram : public OclProgram<BackpropProgram> {
    static inline std::string program_name() {
      return "catnip_backprop";
    }

    static inline std::string relative_path() {
      return "/trainer/backprop.cl";
    }

    static inline std::string* kernel_names(unsigned int& num_kernels) {
      num_kernels = 1;
      static std::string kernel_names[1] = { "backprop_update_local"};
      return kernel_names;
    }
  };

  class BackpropTrainer {
    public:
      struct Delegate {
        virtual void update(
            Layer& layer,
            viennacl::vector<float>& sigma) = 0;

        virtual void update(
            Connection& connection, 
            viennacl::vector<float>& sigma,
            viennacl::vector<float>& inputs) = 0;

        virtual void backpropogate(
            Layer& prev_layer, 
            Connection& connection, 
            Layer& layer, 
            viennacl::vector<float>& prev_derivative) = 0;
      };

      BackpropTrainer(float alpha);
      BackpropTrainer(Delegate* delegate);
      
      void train(FeedForwardNetwork& network, LabeledDataset& ds, int epochs);

    private:
      boost::scoped_ptr<Delegate> delegate_;
      
      boost::ptr_vector<Layer> sigmas_;
      boost::ptr_vector<viennacl::vector<float> > outputs_;
      boost::ptr_vector<viennacl::vector<float> > derivatives_;
  };

  struct DefaultDelegate : public BackpropTrainer::Delegate {
    DefaultDelegate(float alpha) : alpha_(alpha) {}

    virtual void update(
        Layer& layer,
        viennacl::vector<float>& sigma);

    virtual void update(
        Connection& connection, 
        viennacl::vector<float>& sigma,
        viennacl::vector<float>& inputs);

    void backpropogate(
        Layer& prev_layer, 
        Connection& connection, 
        Layer& layer, 
        viennacl::vector<float>& prev_derivative);

    viennacl::scalar<float> alpha_;
  };
}

#endif // BACKPROP_H_