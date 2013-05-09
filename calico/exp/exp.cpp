#define VIENNACL_WITH_OPENCL
//#define VIENNACL_BUILD_INFO

#include <signal.h>
#include <time.h>
#include <algorithm>
#include <math.h>
#include <boost/array.hpp>
#include <boost/scoped_ptr.hpp>

#include "activation/activation.h"
#include "layer/layer.h"
#include "layer/feature.h"
#include "connection/convolution.h"
#include "network/init.h"
#include "network/feedforward/feedforward.h"
#include "network/rbm/rbm.h"
#include "trainer/backprop.h"

#include "data/cifar.h"

using namespace catnip;

inline float round(float r) {
  return (r > 0.0f) ? floor(r + 0.5f) : ceil(r - 0.5f);
}

void cifar() {
  ConvolutionProgram::init();
  viennacl::ocl::set_context_device_type(0, viennacl::ocl::cpu_tag());

  CifarDataset ds(3 * 1024, 10);
  CifarDataset ds2(3 * 1024, 10);
  ds.load("/Users/ylou/Code/catnip/res/cifar-10-batches-bin/data_batch_2.bin", 100);
  ds2.load("/Users/ylou/Code/catnip/res/cifar-10-batches-bin/test_batch.bin", 100);

  LinearActivator scaling_activator(1.0f / 256.0f);
  SoftmaxActivator softmax_activator;
  TanhActivator activator;

  boost::scoped_ptr<FeedForwardNetwork> network(new FeedForwardNetwork());
  network->set_input_layer(new FeatureMapLayer(scaling_activator, 32, 32, 3));
  network->add_layer(
      new ConvolutionConnection(32, 32, 3, 24, 5, 5), 
      new FeatureMapLayer(activator, 28, 28, 24));
  network->add_layer(
      new MaxPoolingConnection(28, 28, 24, 2),
      new FeatureMapLayer(activator, 14, 14, 24));
  network->add_layer(
      new ConvolutionConnection(14, 14, 24, 3, 5, 5), 
      new FeatureMapLayer(activator, 10, 10, 3));
  network->add_layer(
      new MaxPoolingConnection(10, 10, 3, 2),
      new FeatureMapLayer(activator, 5, 5, 3));
  network->add_layer(
      new FullConnection(5 * 5 * 3, 10), 
      new Layer(softmax_activator, 10));

  FanInInitializer ri(1.0f);
  network->initialize(ri);

  BackpropTrainer trainer(0.5f);
  trainer.train(*network, ds, 200);

  int correct = 0;
  boost::array<float, 10> guess;
  boost::array<float, 10> actual;
  for (int i = 0; i < ds2.size(); ++i) {
    network->activate(ds2.get_input(i));
    
    fast_copy(network->get_output_layer().get_value(), guess);
    fast_copy(ds2.get_output(i), actual);

    int guess_class = (int) std::distance(guess.begin(), std::max_element(guess.begin(), guess.end()));
    int actual_class = (int) std::distance(actual.begin(), std::max_element(actual.begin(), actual.end()));

    if (guess_class == actual_class) {
      DEBUG(guess_class << ":" << actual_class);
      correct++;
    }
  }

  std::cout << "Percent correct=" << (100.0f * correct) / ds2.size() << std::endl;
}

void ff_784_10(LabeledDataset& ds) {
  LinearActivator scaling_activator(1.0f);
  TanhActivator activator;

  boost::scoped_ptr<FeedForwardNetwork> network(new FeedForwardNetwork());
  network->set_input_layer(new Layer(scaling_activator, 784));
  network->add_layer(new FullConnection(784, 10), new BiasedLayer(activator, 10));

  RandomInitializer ri(0.1f);
  network->initialize(ri);

  BackpropTrainer trainer(0.1f);
  trainer.train(*network, ds, 1);
}

void ff_784_500_10(LabeledDataset& ds) {
  LinearActivator scaling_activator(1.0f);
  TanhActivator activator;

  boost::scoped_ptr<FeedForwardNetwork> network(new FeedForwardNetwork());
  network->set_input_layer(new Layer(scaling_activator, 784));
  network->add_layer(new FullConnection(784, 500), new BiasedLayer(activator, 500));
  network->add_layer(new FullConnection(500, 10), new BiasedLayer(activator, 10));

  RandomInitializer ri(0.1f);
  network->initialize(ri);

  BackpropTrainer trainer(0.1f);
  trainer.train(*network, ds, 1);
}

void ff_784_1000_1000_1000_10(LabeledDataset& ds) {
  LinearActivator scaling_activator(1.0f);
  TanhActivator activator;

  boost::scoped_ptr<FeedForwardNetwork> network(new FeedForwardNetwork());
  network->set_input_layer(new Layer(scaling_activator, 784));
  network->add_layer(new FullConnection(784, 1000), new BiasedLayer(activator, 1000));
  network->add_layer(new FullConnection(1000, 1000), new BiasedLayer(activator, 1000));
  network->add_layer(new FullConnection(1000, 1000), new BiasedLayer(activator, 1000));
  network->add_layer(new FullConnection(1000, 10), new BiasedLayer(activator, 10));

  RandomInitializer ri(0.1f);
  network->initialize(ri);

  BackpropTrainer trainer(0.1f);
  trainer.train(*network, ds, 1);
}

#include <windows.h>

long long ms_now() {
    static LARGE_INTEGER s_frequency;
    static BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
    if (s_use_qpc) {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return (1000LL * now.QuadPart) / s_frequency.QuadPart;
    } else {
        return GetTickCount();
    }
}

int main() {
  srand((unsigned int) time(0));

  cifar();

  //LabeledDataset ds(784, 10);
  //for (unsigned int i = 0; i < 6000; ++i) {
  //  boost::array<float, 784> ia;
  //  for (unsigned int j = 0; j < 784; ++j) {
  //    ia[j] = rand();
  //  }

  //  boost::array<float, 10> oa;
  //  for (unsigned int j = 0; j < 10; ++j) {
  //    oa[j] = rand();
  //  }

  //  ds.add_observation(ia, oa);
  //}

  //ff_784_10(ds);

  //long long now = ms_now();
  //ff_784_10(ds);
  //std::cout << 6000 * 1000.0 / (ms_now() - now) << std::endl;

  //now = ms_now();
  //ff_784_500_10(ds);
  //std::cout << 6000 * 1000.0 / (ms_now() - now) << std::endl;
  //
  //now = ms_now();
  //ff_784_1000_1000_1000_10(ds);
  //std::cout << 6000 * 1000.0 / (ms_now() - now) << std::endl;

  system("pause");
}