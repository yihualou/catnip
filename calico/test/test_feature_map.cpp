#include <gtest/gtest.h>

#include "activation/activation.h"
#include "layer/feature.h"
#include "connection/convolution.h"

using namespace catnip;

TEST(FeatureMapLayer, Constructor) {  
  SigmoidActivator activator;
  FeatureMapLayer layer(activator, 5, 5, 5);
  
  FeatureMapArray::MatrixRange& fm1 = layer.get_feature_map(0);
  fm1(0, 0) = 1337.0f;
  
  FeatureMapArray::MatrixRange& fm2 = layer.get_feature_map(1);
  fm2(0, 0) = 1234.0f;

  std::vector<float> host_vector(layer.get_value().size());
  fast_copy(layer.get_value(), host_vector);

  EXPECT_NEAR(host_vector[0], 1337, 1e-4);
  EXPECT_NEAR(host_vector[5 * 5], 1234, 1e-4);
}