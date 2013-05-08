#ifndef MNIST_H_
#define MNIST_H_

#include "global_defines.h"

#include "data/labeled.h"

namespace catnip {
  class MnistDataset : public LabeledDataset {
    public:
      void load(const std::string& filename);
  };
}

#endif // MNIST_sH_
