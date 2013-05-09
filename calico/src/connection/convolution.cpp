#include "global_defines.h"

#include "layer/feature.h"
#include "convolution.h"

namespace catnip {
  std::map<cl_context, bool> ConvolutionProgram::init_done;

  ConvolutionConnection::ConvolutionConnection(
      unsigned int width,
      unsigned int height,
      unsigned int num_input_feature_maps, 
      unsigned int num_output_feature_maps, 
      unsigned int filter_width, 
      unsigned int filter_height)
      : Connection(width * height * num_input_feature_maps, calc_output_width(width, filter_width) * calc_output_height(height, filter_height) * num_output_feature_maps), 
        width_(width), height_(height), 
        output_width_(calc_output_width(width, filter_width)), output_height_(calc_output_height(height, filter_height)),
        num_input_feature_maps_(num_input_feature_maps), num_output_feature_maps_(num_output_feature_maps),
        filter_width_(filter_width), filter_height_(filter_height) {

    for (unsigned int i = 0; i < num_input_feature_maps_; ++i) {
      boost::ptr_vector< viennacl::matrix<float> >* row = new boost::ptr_vector< viennacl::matrix<float> >();
      filters_.push_back(row);
      for (unsigned int j = 0; j < num_output_feature_maps_; ++j) {
        viennacl::matrix<float>* mat = new viennacl::matrix<float>(filter_height, filter_width);
        *mat = viennacl::zero_matrix<float>(filter_height, filter_width);
        row->push_back(mat);
      }
    }
  }

  ConvolutionConnection::ConvolutionConnection(const ConvolutionConnection& other)
      : Connection(other.ki_, other.ko_),
        width_(other.width_), height_(other.height_), 
        output_width_(other.output_width_), output_height_(other.output_height_),
        num_input_feature_maps_(other.num_input_feature_maps_), num_output_feature_maps_(other.num_output_feature_maps_),
        filter_width_(other.filter_width_), filter_height_(other.filter_height_) {

    filters_ = other.filters_;
  }

  ConvolutionConnection& ConvolutionConnection::operator=(ConvolutionConnection other) {
    ki_ = other.ki_;
    ko_ = other.ko_;

    width_ = other.width_;
    height_ = other.height_;
    output_width_ = other.output_width_;
    output_height_ = other.output_height_;
    filter_width_ = other.filter_width_;
    filter_height_ = other.filter_height_;
    filters_ = other.filters_;
    
    return *this;
  }

  ConvolutionConnection* ConvolutionConnection::clone() const {
    return new ConvolutionConnection(*this);
  }

  std::ostream& ConvolutionConnection::put(std::ostream& stream) {
    stream << "C";
    return stream;
  }

  viennacl::matrix<float>& ConvolutionConnection::get_filter(const int i, const int j) {
    return filters_[i][j];
  }

  void ConvolutionConnection::propogate(viennacl::vector<float>& input, viennacl::vector<float>& output) {
    FeatureMapArray input_arr(input, width_, height_, num_input_feature_maps_);
    FeatureMapArray output_arr(output, output_width_, output_height_, num_output_feature_maps_);

    for (unsigned int j = 0; j < num_output_feature_maps_; ++j) {
      FeatureMapArray::MatrixRange& fmo = output_arr[j];

      for (unsigned int i = 0; i < num_input_feature_maps_; ++i) {
        viennacl::matrix<float> buffer = viennacl::zero_matrix<float>(output_height_, output_width_);

        FeatureMapArray::MatrixRange& fmi = input_arr[i];
        convolve2d_valid(fmi, buffer, filters_[i][j], width_, height_, filter_width_, filter_height_);

        fmo += buffer;
      }
    }
  }

  unsigned int ConvolutionConnection::calc_output_width(unsigned int width, unsigned int filter_width) {
    return width - filter_width + 1;
  }

  unsigned int ConvolutionConnection::calc_output_height(unsigned int height, unsigned int filter_height) {
    return height - filter_height + 1;
  }

  MaxPoolingConnection::MaxPoolingConnection(
      unsigned int width,
      unsigned int height,
      unsigned int num_feature_maps, 
      unsigned int factor)
      : Connection(width * height * num_feature_maps, (width * height * num_feature_maps) / (factor * factor)),
        width_(width), height_(height), 
        output_width_(width / factor), output_height_(height / factor),
        num_feature_maps_(num_feature_maps), factor_(factor) {

      for (int i = 0; i < num_feature_maps_; ++i) {
        betas_.push_back(new viennacl::scalar<float>(0.0f));
      }
  }


  MaxPoolingConnection::MaxPoolingConnection(const MaxPoolingConnection& other) 
      : Connection(other.ki_, other.ko_),
        width_(other.width_), height_(other.height_), 
        output_width_(other.output_width_), output_height_(other.output_height_),
        num_feature_maps_(other.num_feature_maps_), factor_(other.factor_) {

      for (int i = 0; i < num_feature_maps_; ++i) {
        betas_.push_back(new viennacl::scalar<float>(0.0f));
      }
  }

  MaxPoolingConnection& MaxPoolingConnection::operator=(MaxPoolingConnection other) {
    ki_ = other.ki_;
    ko_ = other.ko_;

    width_ = other.width_;
    height_ = other.height_;
    output_width_ = other.output_width_;
    output_height_ = other.output_height_;
    num_feature_maps_ = other.num_feature_maps_;
    factor_ = other.factor_;
    betas_ = other.betas_;
    
    return *this;
  }

  MaxPoolingConnection* MaxPoolingConnection::clone() const {
    return new MaxPoolingConnection(*this);
  }

  std::ostream& MaxPoolingConnection::put(std::ostream& stream) {
    stream << "C";
    return stream;
  }

  void MaxPoolingConnection::propogate(viennacl::vector<float>& input, viennacl::vector<float>& output) {
    //ConvolutionProgram::init();

    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ConvolutionProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("maxpool2d");

    k.local_work_size(0, 1);
    k.global_work_size(0, 128);
    k.local_work_size(1, 1);
    k.global_work_size(1, 128);

    viennacl::ocl::enqueue(
      k(
        input,
        output,
        width_,
        height_,
        factor_
      )
    );
  }

  void convolve2d_full(
      viennacl::matrix_base<float>& input,
      viennacl::matrix_base<float>& output, 
      viennacl::matrix_base<float>& filter,
      unsigned int width,
      unsigned int height,
      unsigned int filter_width, 
      unsigned int filter_height) {

    //ConvolutionProgram::init();

    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ConvolutionProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("convolve2d_full");

    k.local_work_size(0, 1);
    k.global_work_size(0, 128);
    k.local_work_size(1, 1);
    k.global_work_size(1, 128);

    size_t local_width = width / k.global_work_size(0) + 3 * filter_width;
    size_t local_height = height / k.global_work_size(1) + 3 * filter_height;

    viennacl::ocl::enqueue(
      k(
        input,
        output,
        filter,
        viennacl::ocl::local_mem(sizeof(float) * local_width * local_height),
        width,
        height,
        filter_width,
        filter_height
      )
    );
  }

  void convolve2d_valid(
      viennacl::matrix_base<float>& input,
      viennacl::matrix_base<float>& output, 
      viennacl::matrix_base<float>& filter,
      unsigned int width,
      unsigned int height,
      unsigned int filter_width, 
      unsigned int filter_height) {

    //ConvolutionProgram::init();

    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ConvolutionProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("convolve2d_valid");

    k.local_work_size(0, 1);
    k.global_work_size(0, 128);
    k.local_work_size(1, 1);
    k.global_work_size(1, 128);

    size_t local_width = width / k.global_work_size(0) + filter_width;
    size_t local_height = height / k.global_work_size(1) + filter_height;

    viennacl::ocl::enqueue(
      k(
        input,
        output,
        filter,
        viennacl::ocl::local_mem(sizeof(float) * local_width * local_height),
        width,
        height,
        filter_width,
        filter_height
      )
    );
  }

  void upsample2d(viennacl::matrix_base<float>& input, viennacl::matrix_base<float>& output, const unsigned int width, const unsigned int height, const unsigned int factor) {
    //ConvolutionProgram::init();

    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ConvolutionProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("upsample2d");

    k.local_work_size(0, 1);
    k.global_work_size(0, 128);
    k.local_work_size(1, 1);
    k.global_work_size(1, 128);

    viennacl::ocl::enqueue(
      k(
        input,
        output,
        width,
        height,
        factor
      )
    );
  }
}