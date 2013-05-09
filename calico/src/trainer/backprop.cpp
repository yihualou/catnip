#include <boost/random.hpp>
#include <viennacl/linalg/prod.hpp>

#include "connection/convolution.h"
#include "layer/feature.h"
#include "backprop.h"

namespace catnip {
  std::map<cl_context, bool> BackpropProgram::init_done;

  BackpropTrainer::BackpropTrainer(float alpha) : delegate_(new DefaultDelegate(alpha)) {}

  BackpropTrainer::BackpropTrainer(Delegate* delegate) : delegate_(delegate) {}

  void BackpropTrainer::train(FeedForwardNetwork& network, LabeledDataset& ds, int epochs) {
    FeedForwardNetwork::LayerIterator lit;
    for (lit = network.layers_begin(); lit != network.layers_end(); ++lit) {
      int k = lit->get_k();

      sigmas_.push_back(lit->clone());
      outputs_.push_back(new viennacl::vector<float>(k));
      derivatives_.push_back(new viennacl::vector<float>(k));
    }

    std::vector<int> indexes;
    for (int i = 0; i < ds.size(); ++i) {
      indexes.push_back(i);
    }

    viennacl::vector<float> error;
    std::vector<float> host_error(ds.output_size(), 0);

    for (int i = 0; i < epochs; ++i) {
      float avg_rms_error = 0.0f;

      std::random_shuffle(indexes.begin(), indexes.end());
      for (std::vector<int>::iterator it = indexes.begin(); it != indexes.end(); ++it) {
        const viennacl::vector<float>& input = ds.get_input(*it);
        const viennacl::vector<float>& output = ds.get_output(*it);

        // Forward pass.
        network.get_input_layer().get_value() = input;
        for (int j = 0; j < network.get_num_connections(); ++j) {
          Layer& layer = network.get_layer(j);
          Layer& next_layer = network.get_layer(j + 1);
          Connection& conn = network.get_connection(j);

          layer.activate(derivatives_[j]);
          conn.layer_propogate(layer, next_layer);
          viennacl::ocl::get_queue().finish();
          
          outputs_[j] = layer.get_value();
        }
        Layer& output_layer = network.get_output_layer();
        output_layer.activate(derivatives_.back());

        // Backwards pass.
        error = output - output_layer.get_value();
        sigmas_.back().get_value() = viennacl::linalg::element_prod(derivatives_.back(), error); 
        for (int j = network.get_num_layers() - 1; j > 0; --j) {
          Layer& prev_layer = network.get_layer(j - 1);
          Layer& layer = network.get_layer(j);
          Connection& conn = network.get_connection(j - 1);

          delegate_->backpropogate(sigmas_[j - 1], conn, sigmas_[j], derivatives_[j - 1]);
          viennacl::ocl::get_queue().finish();

          delegate_->update(conn, sigmas_[j].get_value(), outputs_[j - 1]);
          viennacl::ocl::get_queue().finish();

          delegate_->update(layer, sigmas_[j].get_value());
          viennacl::ocl::get_queue().finish();
        } 

        fast_copy(error, host_error);

        float rms_error = 0.0f;
        for (size_t j = 0; j < host_error.size(); ++j) {
          rms_error += host_error[j] * host_error[j];
        }

        rms_error = sqrt(rms_error / host_error.size());
        avg_rms_error += rms_error;
      }

      avg_rms_error /= ds.size();

      std::cout << "Epoch=" << i << " Avg-RMS=" << avg_rms_error << std::endl;
    }
  }

  void DefaultDelegate::update(
      Layer& layer,
      viennacl::vector<float>& sigma) {

    if (BiasedLayer* dl = dynamic_cast<BiasedLayer*>(&layer)) {
      dl->get_biases() += alpha_ * sigma;
    } else if (FeatureMapLayer* dl = dynamic_cast<FeatureMapLayer*>(&layer)) {
      FeatureMapArray arr(sigma, dl->get_width(), dl->get_height(), dl->get_num_feature_maps());
      for (int i = 0; i < dl->get_num_feature_maps(); ++i) {
        dl->get_bias(i) += alpha_ * arr[i];
      }
    }
  }

  void DefaultDelegate::update(
      Connection& connection, 
      viennacl::vector<float>& sigma,
      viennacl::vector<float>& inputs) {

    if (FullConnection* dc = dynamic_cast<FullConnection*>(&connection)) {
      viennacl::linalg::scaled_rank_1_update(dc->get_weights(), alpha_, 1, false, false, sigma, inputs);
    } else if (IdentityConnection* dc = dynamic_cast<IdentityConnection*>(&connection)) {
      dc->get_weights() += alpha_ * viennacl::linalg::element_prod(sigma, inputs);
    } else if (ConvolutionConnection* dc = dynamic_cast<ConvolutionConnection*>(&connection)) {
      FeatureMapArray inputs_arr(
        inputs,
        dc->get_width(),
        dc->get_height(),
        dc->get_num_input_feature_maps());
      
      FeatureMapArray sigma_arr(
        sigma,
        dc->get_output_width(), 
        dc->get_output_height(), 
        dc->get_num_output_feature_maps());

      for (int i = 0; i < dc->get_num_input_feature_maps(); ++i) {
        for (int j = 0; j < dc->get_num_output_feature_maps(); ++j) {
          viennacl::matrix<float> buffer(dc->get_filter_width(), dc->get_filter_height());
          viennacl::matrix<float> sigma_trans = trans(sigma_arr[j]);
          convolve2d_valid(
            inputs_arr[i], 
            buffer,
            sigma_trans,
            dc->get_output_width(),
            dc->get_output_height(),
            dc->get_filter_width(),
            dc->get_filter_height());

          viennacl::matrix<float> buffer_trans = trans(buffer);
          dc->get_filter(i, j) += alpha_ * buffer_trans;
        }
      }
    } else if (MaxPoolingConnection* dc = dynamic_cast<MaxPoolingConnection*>(&connection)) {
    
    }
  }

  void DefaultDelegate::backpropogate(
      Layer& prev_layer, 
      Connection& connection, 
      Layer& layer, 
      viennacl::vector<float>& prev_derivative) {

    if (FullConnection* dc = dynamic_cast<FullConnection*>(&connection)) {
      prev_layer.get_value() = viennacl::linalg::element_prod(
        viennacl::linalg::prod(trans(dc->get_weights()), layer.get_value()),
        prev_derivative);

    } else if (IdentityConnection* dc = dynamic_cast<IdentityConnection*>(&connection)) {
      prev_layer.get_value() = viennacl::linalg::element_prod(
        viennacl::linalg::element_prod(dc->get_weights(), layer.get_value()),
        prev_derivative);

    } else if (ConvolutionConnection* dc = dynamic_cast<ConvolutionConnection*>(&connection)) {
      FeatureMapLayer* fml_prev = dynamic_cast<FeatureMapLayer*>(&prev_layer);
      FeatureMapLayer* fml_curr = dynamic_cast<FeatureMapLayer*>(&layer);

      for (int i = 0; i < fml_prev->get_num_feature_maps(); ++i) {
        FeatureMapArray::MatrixRange& fm_prev = fml_prev->get_feature_map(i);
        viennacl::matrix<float> sigma_buffer = 
          viennacl::zero_matrix<float>(fm_prev.size1(), fm_prev.size2());

        for (int j = 0; j < fml_curr->get_num_feature_maps(); ++j) {
          FeatureMapArray::MatrixRange& fm_curr = fml_curr->get_feature_map(j);

          // Overwrite fm_prev and save sum in sigma_buffer
          viennacl::matrix<float> filter_trans = trans(dc->get_filter(i, j));
          convolve2d_full(
            fm_curr, 
            fm_prev,
            filter_trans,
            fml_curr->get_width(),
            fml_curr->get_height(),
            dc->get_filter_width(),
            dc->get_filter_height());

          sigma_buffer += fm_prev;
        }
        fm_prev = sigma_buffer;
      }

      fml_prev->get_value() = viennacl::linalg::element_prod(
        fml_prev->get_value(),
        prev_derivative);

    } else if (MaxPoolingConnection* dc = dynamic_cast<MaxPoolingConnection*>(&connection)) {
      FeatureMapLayer* fml_prev = dynamic_cast<FeatureMapLayer*>(&prev_layer);
      FeatureMapLayer* fml_curr = dynamic_cast<FeatureMapLayer*>(&layer);
      
      // fml_prev and fml_curr have the same # of feature maps
      for (int i = 0; i < fml_prev->get_num_feature_maps(); ++i) {
        FeatureMapArray::MatrixRange& fm_prev = fml_prev->get_feature_map(i);
        FeatureMapArray::MatrixRange& fm_curr = fml_curr->get_feature_map(i);
        
        upsample2d(
          fm_curr, 
          fm_prev,
          fml_curr->get_width(),
          fml_curr->get_height(),
          dc->get_factor());
      }

      fml_prev->get_value() = viennacl::linalg::element_prod(
        fml_prev->get_value(),
        prev_derivative);

      for (int i = 0; i < fml_prev->get_num_feature_maps(); ++i) {
        FeatureMapArray::MatrixRange& fm_prev = fml_prev->get_feature_map(i);
        fm_prev *= dc->get_betas()[i];
      }
    }
  }
}
