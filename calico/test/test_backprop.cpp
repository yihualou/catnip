#include "global_defines.h"

#include <time.h>
#include <stdlib.h>
#include <gtest/gtest.h>
#include <boost/array.hpp>
#include <boost/scoped_ptr.hpp>

#include "activation/activation.h"
#include "layer/layer.h"
#include "network/init.h"
#include "network/feedforward/feedforward.h"
#include "network/rbm/rbm.h"
#include "trainer/backprop.h"

using namespace catnip;

TEST(BackpropTrainer, XorTraining) {
  srand(0);

  LinearActivator linear_activator(1.0f);
  SigmoidActivator activator;

  FullConnection* c1 = new FullConnection(2, 2);
  FullConnection* c2 = new FullConnection(2, 1);

  Layer* l1 = new Layer(linear_activator, 2);
  BiasedLayer* l2 = new BiasedLayer(activator, 2);
  BiasedLayer* l3 = new BiasedLayer(activator, 1);

  FeedForwardNetwork network;
  network.set_input_layer(l1);
  network.add_layer(c1, l2);
  network.add_layer(c2, l3);

  LabeledDataset ds(2, 1);
  boost::array<float, 2> i1 = { 0.0f, 0.0f };
  boost::array<float, 2> i2 = { 0.0f, 1.0f };
  boost::array<float, 2> i3 = { 1.0f, 0.0f };
  boost::array<float, 2> i4 = { 1.0f, 1.0f };

  boost::array<float, 1> o1 = { 0.0f };
  boost::array<float, 1> o2 = { 1.0f };
  boost::array<float, 1> o3 = { 1.0f };
  boost::array<float, 1> o4 = { 1.0f };

  ds.add_observation(i1, o1);
  ds.add_observation(i2, o2);
  ds.add_observation(i3, o3);
  ds.add_observation(i4, o4);

  RandomInitializer ri(1.0f);
  network.initialize(ri);

  BackpropTrainer trainer(0.5f);
  trainer.train(network, ds, 2000);

  for (int i = 0; i < 4; ++i) {
    network.activate(ds.get_input(i));
    float guess = network.get_output_layer().get_value()[0];
    float actual = ds.get_output(i)[0];
    EXPECT_NEAR(guess, actual, 0.1); 
  }
}