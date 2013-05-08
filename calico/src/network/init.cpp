#include <stdlib.h>
#include <time.h>

#include "init.h"

namespace catnip {
  RandomInitializer::RandomInitializer(float magnitude) : magnitude_(magnitude) {
    InitializerProgram::init(); 
  }

  void RandomInitializer::initialize(Layer& layer) const {
    if (BiasedLayer* dl = dynamic_cast<BiasedLayer*>(&layer)) {
      viennacl::vector<float>& vector = dl->get_biases();

      viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
      viennacl::ocl::kernel& k = prog.get_kernel("random_initialize");
      viennacl::ocl::enqueue(k(vector, cl_uint(vector.size()), magnitude_, cl_uint(rand())));
    }
  }

  void RandomInitializer::initialize(Connection& connection) const {
    if (FullConnection* dc = dynamic_cast<FullConnection*>(&connection)) {
      viennacl::matrix<float>& matrix = dc->get_weights();

      viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
      viennacl::ocl::kernel& k = prog.get_kernel("random_initialize");
      viennacl::ocl::enqueue(k(matrix, cl_uint(matrix.size1() * matrix.size2()), magnitude_, cl_uint(rand())));

    } else if (IdentityConnection* dc = dynamic_cast<IdentityConnection*>(&connection)) {
      viennacl::vector<float>& vector = dc->get_weights();

      viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
      viennacl::ocl::kernel& k = prog.get_kernel("random_initialize");
      viennacl::ocl::enqueue(k(vector, cl_uint(vector.size()), magnitude_, cl_uint(rand())));
    } else if (SparseConnection* dc = dynamic_cast<SparseConnection*>(&connection)) {
      viennacl::compressed_matrix<float>& matrix = dc->get_weights();

      viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
      viennacl::ocl::kernel& k = prog.get_kernel("random_initialize");
      viennacl::ocl::enqueue(k(matrix, cl_uint(matrix.size1() * matrix.size2()), magnitude_, cl_uint(rand())));
    }
  }

  FanInInitializer::FanInInitializer(float magnitude) : magnitude_(magnitude) {
    InitializerProgram::init();
  }

  void FanInInitializer::initialize(Layer& layer) const {
    if (BiasedLayer* dl = dynamic_cast<BiasedLayer*>(&layer)) {
      initialize(*dl);
    }
  }

  void FanInInitializer::initialize(Connection& connection) const {
    if (FullConnection* dc = dynamic_cast<FullConnection*>(&connection)) {
      initialize(*dc);
    } else if (IdentityConnection* dc = dynamic_cast<IdentityConnection*>(&connection)) {
      initialize(*dc);
    } else if (SparseConnection* dc = dynamic_cast<SparseConnection*>(&connection)) {
      initialize(*dc);
    } else if (ConvolutionConnection* dc = dynamic_cast<ConvolutionConnection*>(&connection)) {
      initialize(*dc);
    } else if (MaxPoolingConnection* dc = dynamic_cast<MaxPoolingConnection*>(&connection)) {
      initialize(*dc);
    }
  }

  void FanInInitializer::initialize(BiasedLayer& layer) const {
    viennacl::vector<float>& vector = layer.get_biases();
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("fan_in_initialize");

    viennacl::ocl::enqueue(
      k(
        vector,
        cl_uint(vector.size()),
        cl_float(magnitude_),
        cl_float(1.0f),
        cl_uint(time(0))
      )
    );
  }

  void FanInInitializer::initialize(FeatureMapLayer& layer) const {
    for (int i = 0; i < layer.get_num_feature_maps(); ++i) {
      viennacl::matrix<float>& matrix = layer.get_bias(i);
      viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
      viennacl::ocl::kernel& k = prog.get_kernel("fan_in_initialize");

      viennacl::ocl::enqueue(
        k(
          matrix,
          cl_uint(matrix.size1() * matrix.size2()),
          cl_float(magnitude_),
          cl_float(1.0f),
          cl_uint(time(0))
        )
      );
    }
  }

  void FanInInitializer::initialize(FullConnection& connection) const {
    viennacl::matrix<float>& matrix = connection.get_weights();
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("fan_in_initialize");

    viennacl::ocl::enqueue(
      k(
        matrix,
        cl_uint(matrix.size1() * matrix.size2()),
        cl_float(magnitude_),
        cl_float(connection.get_ki()),
        cl_uint(time(0))
      )
    );
  }

  void FanInInitializer::initialize(IdentityConnection& connection) const {
    viennacl::vector<float>& vector = connection.get_weights();
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("fan_in_initialize");

    viennacl::ocl::enqueue(
      k(
        vector,
        cl_uint(vector.size()),
        cl_float(magnitude_),
        cl_float(connection.get_ki()),
        cl_uint(time(0))
      )
    );
  };

  void FanInInitializer::initialize(SparseConnection& connection) const {
    viennacl::compressed_matrix<float>& matrix = connection.get_weights();
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("random_initialize");
    viennacl::ocl::enqueue(k(matrix, cl_uint(matrix.size1()), magnitude_, cl_uint(time(0))));
  };

  void FanInInitializer::initialize(ConvolutionConnection& connection) const {
    for (int i = 0; i < connection.get_num_input_feature_maps(); ++i) {
      for (int j = 0; j < connection.get_num_output_feature_maps(); ++j) {
        viennacl::matrix<float>& matrix = connection.get_filter(i, j);
        viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
        viennacl::ocl::kernel& k = prog.get_kernel("fan_in_initialize");

        viennacl::ocl::enqueue(
          k(
            matrix,
            cl_uint(matrix.size1() * matrix.size2()),
            cl_float(magnitude_),
            cl_float(matrix.size2()),
            cl_uint(time(0))
          )
        );
      }
    }
  };

  void FanInInitializer::initialize(MaxPoolingConnection& connection) const {
    for (int i = 0; i < connection.get_betas().size(); ++i) {
      viennacl::scalar<float>& scalar = connection.get_betas()[i];
      viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(InitializerProgram::program_name());
      viennacl::ocl::kernel& k = prog.get_kernel("fan_in_initialize");

      viennacl::ocl::enqueue(
        k(
          scalar,
          cl_uint(1),
          cl_float(magnitude_),
          cl_float(1.0f),
          cl_uint(time(0))
        )
      );
    }
  };
}