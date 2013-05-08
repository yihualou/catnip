#include "gtest/gtest.h"

#include "activation/activation.h"

using namespace catnip;

TEST(LinearActivator, Activation) {
  LinearActivator activator(2.0f);
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 42.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -100.0f);
  viennacl::vector<float> r1 = viennacl::scalar_vector<float>(5, 84.0f);
  viennacl::vector<float> r2 = viennacl::scalar_vector<float>(5, -200.0f);

  activator.activate(v1, v1);
  activator.activate(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r1[i], 1e-4f);
    EXPECT_NEAR(v2[i], r2[i], 1e-4f);
  }
}

TEST(LinearActivator, Derivatives) {
  LinearActivator activator(2.0f);
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 42.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -100.0f);
  viennacl::vector<float> r = viennacl::scalar_vector<float>(5, 2.0f);

  activator.derivatives(v1, v1);
  activator.derivatives(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r[i], 1e-4f);
    EXPECT_NEAR(v2[i], r[i], 1e-4f);
  }
}

TEST(SigmoidActivator, Activation) {
  SigmoidActivator activator;
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 1.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -0.5f);
  viennacl::vector<float> r1 = viennacl::scalar_vector<float>(5, 0.7310585f);
  viennacl::vector<float> r2 = viennacl::scalar_vector<float>(5, 0.3775407f);

  activator.activate(v1, v1);
  activator.activate(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r1[i], 1e-4f);
    EXPECT_NEAR(v2[i], r2[i], 1e-4f);
  }
}

TEST(SigmoidActivator, Derivatives) {
  SigmoidActivator activator;
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 1.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -0.5f);
  viennacl::vector<float> r1 = viennacl::scalar_vector<float>(5, 0.196612f);
  viennacl::vector<float> r2 = viennacl::scalar_vector<float>(5, 0.235004f);

  activator.derivatives(v1, v1);
  activator.derivatives(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r1[i], 1e-4f);
    EXPECT_NEAR(v2[i], r2[i], 1e-4f);
  }
}

TEST(TanhActivator, Activation) {
  TanhActivator activator;
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 1.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -0.5f);
  viennacl::vector<float> r1 = viennacl::scalar_vector<float>(5, 0.761594f);
  viennacl::vector<float> r2 = viennacl::scalar_vector<float>(5, -0.4621177f);

  activator.activate(v1, v1);
  activator.activate(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r1[i], 1e-4f);
    EXPECT_NEAR(v2[i], r2[i], 1e-4f);
  }
}

TEST(TanhActivator, Derivatives) {
  TanhActivator activator;
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 1.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -0.5f);
  viennacl::vector<float> r1 = viennacl::scalar_vector<float>(5, 0.419974f);
  viennacl::vector<float> r2 = viennacl::scalar_vector<float>(5, 0.786448f);

  activator.derivatives(v1, v1);
  activator.derivatives(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r1[i], 1e-4f);
    EXPECT_NEAR(v2[i], r2[i], 1e-4f);
  }
}

TEST(SoftmaxActivator, Activation) {
  SoftmaxActivator activator;
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 1.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -0.5f);
  viennacl::vector<float> r1 = viennacl::scalar_vector<float>(5, 1.0f / 5.0f);
  viennacl::vector<float> r2 = viennacl::scalar_vector<float>(5, 1.0f / 5.0f);

  activator.activate(v1, v1);
  activator.activate(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r1[i], 1e-4f);
    EXPECT_NEAR(v2[i], r2[i], 1e-4f);
  }
}

TEST(SoftmaxActivator, Derivatives) {
  SoftmaxActivator activator;
  
  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(5, 1.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(5, -0.5f);
  viennacl::vector<float> r1 = viennacl::scalar_vector<float>(5, 1.0f / 5.0f);
  viennacl::vector<float> r2 = viennacl::scalar_vector<float>(5, 1.0f / 5.0f);

  activator.derivatives(v1, v1);
  activator.derivatives(v2, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_NEAR(v1[i], r1[i] * (1 - r1[i]), 1e-4f);
    EXPECT_NEAR(v2[i], r2[i] * (1 - r2[i]), 1e-4f);
  }
}


