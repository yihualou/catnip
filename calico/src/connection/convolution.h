#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "global_defines.h"

#include <boost/ptr_container/ptr_vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/compressed_matrix.hpp>

#include "layer/layer.h"
#include "connection.h"

namespace catnip {
  struct ConvolutionProgram : public OclProgram<ConvolutionProgram> {
    static inline std::string program_name() {
      return "catnip_convolution";
    }

    static inline std::string relative_path() {
      return "/connection/convolution.cl";
    }

    static inline std::string* kernel_names(unsigned int& num_kernels) {
      num_kernels = 4;
      static std::string kernel_names[4] = { "convolve2d_valid", "convolve2d_full", "maxpool2d", "upsample2d" };
      return kernel_names;
    }
  };

  class ConvolutionConnection : public Connection {
    public:
      ConvolutionConnection(
          unsigned int width,
          unsigned int height,
          unsigned int num_input_feature_maps, 
          unsigned int num_output_feature_maps, 
          unsigned int filter_width, 
          unsigned int filter_height);
      ConvolutionConnection(const ConvolutionConnection& other);
      ~ConvolutionConnection() {}

      ConvolutionConnection& operator=(ConvolutionConnection other);
      ConvolutionConnection* clone() const;
      std::ostream& put(std::ostream& stream);

      unsigned int get_width() { return width_; }
      unsigned int get_height() { return height_; }
      unsigned int get_output_width() { return output_width_; }
      unsigned int get_output_height() { return output_height_; }
      unsigned int get_num_input_feature_maps() { return num_input_feature_maps_; }
      unsigned int get_num_output_feature_maps() { return num_output_feature_maps_; }
      unsigned int get_filter_width() { return filter_width_; }
      unsigned int get_filter_height() { return filter_height_; }
      viennacl::matrix<float>& get_filter(const int i, const int j);

      void propogate(viennacl::vector<float>& input, viennacl::vector<float>& output);

      static unsigned int calc_output_width(unsigned int width, unsigned int filter_width);
      static unsigned int calc_output_height(unsigned int height, unsigned int filter_height);

    protected:
      unsigned int width_, height_;
      unsigned int output_width_, output_height_;
      unsigned int num_input_feature_maps_, num_output_feature_maps_;
      unsigned int filter_width_, filter_height_;
      boost::ptr_vector< boost::ptr_vector< viennacl::matrix<float> > > filters_;
  };

  class MaxPoolingConnection : public Connection {
    public:
      MaxPoolingConnection(
          unsigned int width,
          unsigned int height,
          unsigned int num_feature_maps, 
          unsigned int factor);

      MaxPoolingConnection(const MaxPoolingConnection& other);
      ~MaxPoolingConnection() {}

      MaxPoolingConnection& operator=(MaxPoolingConnection other);
      MaxPoolingConnection* clone() const;
      std::ostream& put(std::ostream& stream);
  
      unsigned int get_width() { return width_; }
      unsigned int get_height() { return height_; }
      unsigned int get_output_width() { return output_width_; }
      unsigned int get_output_height() { return output_height_; }
      unsigned int get_num_feature_maps() { return num_feature_maps_; }
      unsigned int get_factor() { return factor_; }
      boost::ptr_vector<viennacl::scalar<float> >& get_betas() { return betas_; }

      void propogate(viennacl::vector<float>& input, viennacl::vector<float>& output);

    private:
      unsigned int width_, height_;
      unsigned int output_width_, output_height_;
      unsigned int num_feature_maps_;
      unsigned int factor_;
      boost::ptr_vector<viennacl::scalar<float> > betas_;
  };

  void convolve2d_full(
    viennacl::matrix_base<float>& input,
    viennacl::matrix_base<float>& output, 
    viennacl::matrix_base<float>& filter,
    unsigned int width,
    unsigned int height,
    unsigned int filter_width, 
    unsigned int filter_height);

  void convolve2d_valid(
    viennacl::matrix_base<float>& input,
    viennacl::matrix_base<float>& output, 
    viennacl::matrix_base<float>& filter,
    unsigned int width,
    unsigned int height,
    unsigned int filter_width, 
    unsigned int filter_height);

  void upsample2d(
    viennacl::matrix_base<float>& input, 
    viennacl::matrix_base<float>& output, 
    const unsigned int width, 
    const unsigned int height,
    const unsigned int factor);
}

#endif