#ifndef CAT_H_
#define CAT_H_

#include "global_defines.h"

#include "data/labeled.h"

namespace catnip {
  class CatDataset : public LabeledDataset {
    public:
      void load(const std::string& filename);
  };
}

#endif // CAT_H_
