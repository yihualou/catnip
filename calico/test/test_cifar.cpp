#include "global_defines.h"

#include <gtest/gtest.h>

#include "data/cifar.h"

using namespace catnip;

TEST(CifarDataset, LoadFile) {
  CifarDataset ds(3 * 1024, 1);
  ds.load("/Users/ylou/Code/catnip/res/cifar-10-batches-bin/data_batch_1.bin", 100);
  // TODO(ylou): Add some asserts here.
}