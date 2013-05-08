#ifndef CONNECTION_H
#define CONNECTION_H

#include "global_defines.h"

#include <viennacl/matrix.hpp>
#include <viennacl/compressed_matrix.hpp>

#include "layer/layer.h"

namespace catnip {
  class Connection {
    public:
      Connection(unsigned int ki, unsigned int ko) : ki_(ki), ko_(ko) {}
      virtual ~Connection() {}

      virtual Connection* clone() const = 0;
      virtual std::ostream& put(std::ostream& stream) = 0;

      unsigned int get_ki();
      unsigned int get_ko();

      virtual void propogate(viennacl::vector<float>& input, viennacl::vector<float>& output) = 0;

      virtual void layer_propogate(Layer& input, Layer& output) {
        propogate(input.get_value(), output.get_value());
      }

    protected:
      unsigned int ki_;
      unsigned int ko_;
  };

  class FullConnection : public Connection {
    public:
      FullConnection(unsigned int ki, unsigned int ko);
      FullConnection(const FullConnection& other);
      ~FullConnection() {}

      FullConnection& operator=(FullConnection other);
      FullConnection* clone() const;
      std::ostream& put(std::ostream& stream);
      
      viennacl::matrix<float>& get_weights() { return weights_; }

      void propogate(viennacl::vector<float>& input, viennacl::vector<float>& output);

    protected:
      viennacl::matrix<float> weights_;
  };

  class IdentityConnection : public Connection {
    public:
      IdentityConnection(unsigned int k);
      IdentityConnection(const IdentityConnection& other);
      ~IdentityConnection() {}

      IdentityConnection& operator=(IdentityConnection other);
      IdentityConnection* clone() const;
      std::ostream& put(std::ostream& stream);

      viennacl::vector<float>& get_weights() { return weights_; }

      void propogate(viennacl::vector<float>& input, viennacl::vector<float>& output);

    protected:
      viennacl::vector<float> weights_;
  };

  class SparseConnection : public Connection {
    public:
      SparseConnection(unsigned int k, const viennacl::compressed_matrix<float>& mask);
      SparseConnection(const SparseConnection& other);
      ~SparseConnection() {}

      SparseConnection& operator=(SparseConnection other);
      SparseConnection* clone() const;
      std::ostream& put(std::ostream& stream);

      viennacl::compressed_matrix<float>& get_weights() { return weights_; }

      void propogate(viennacl::vector<float>& input, viennacl::vector<float>& output);

    protected:
      viennacl::compressed_matrix<float> weights_;
  };

  Connection* new_clone(Connection const& other);

  std::ostream& operator<<(std::ostream& stream, Connection& connection);
}

#endif