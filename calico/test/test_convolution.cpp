#define ENABLE_UBLAS 1

#include <gtest/gtest.h>
#include <boost/numeric/ublas/matrix.hpp>

#include "connection/convolution.h"

using namespace boost::numeric;
using namespace catnip;

TEST(ConvolutionConnection, Convolution) {  
  ConvolutionProgram::init();

  unsigned int in_width = 5, in_height = 5, filter_width = 3, filter_height = 3;
  unsigned int out_width = ConvolutionConnection::calc_output_width(in_width, filter_width);
  unsigned int out_height = ConvolutionConnection::calc_output_height(in_height, filter_height);

  viennacl::vector<float> v1 = viennacl::scalar_vector<float>(in_width * in_height * 4, 1.0f);
  viennacl::vector<float> v2 = viennacl::scalar_vector<float>(out_width * out_height * 4, 0.0f);

  ublas::matrix<float> filter(filter_width, filter_height);
  for (unsigned int i = 0; i < filter.size1(); ++i) {
    for (unsigned int j = 0; j < filter.size2(); ++j) {
      filter(i, j) = -1;
    }
  }
  filter(1, 1) = 5;

  viennacl::matrix<float> vienna_filter(filter_width, filter_height);
  copy(filter, vienna_filter);

  ConvolutionConnection connection(in_width, in_height, 4, 4, filter_width, filter_height);
  connection.get_filter(0, 0) = vienna_filter;
  connection.get_filter(1, 1) = vienna_filter;
  connection.propogate(v1, v2);
  
  for (size_t i = 0; i < 2 * (out_width * out_height); ++i) {
    EXPECT_NEAR(v2[i], -3.0f, 1e-4f);
  }
  for (size_t i = 2 * (out_width * out_height); i < v2.size(); ++i) {
    EXPECT_NEAR(v2[i], 0.0f, 1e-4f);
  }
}