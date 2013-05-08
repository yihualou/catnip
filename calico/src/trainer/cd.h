#ifndef CD_H_
#define CD_H_

#include "network/rbm/rbm.h"

namespace catnip {
  class ContrastiveDivergenceTrainer {
    public:
      ContrastiveDivergenceTrainer(RbmNetwork& network) : network_(network) {}

      virtual void train(const viennacl::vector<float>& input, const viennacl::vector<float>& output);
    
    private:
      RbmNetwork& network_;
  };
}

#endif // CD_H_