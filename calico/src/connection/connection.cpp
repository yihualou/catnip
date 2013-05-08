#include "global_defines.h"

#include <viennacl/linalg/prod.hpp>

#include "connection.h"

namespace catnip {
  unsigned int Connection::get_ki() {
    return ki_;
  }

  unsigned int Connection::get_ko() {
    return ko_;
  }

  FullConnection::FullConnection(unsigned int ki, unsigned int ko) : Connection(ki, ko) {
    weights_ = viennacl::zero_matrix<float>(ko_, ki_);
  }

  FullConnection::FullConnection(const FullConnection& other) : Connection(other.ki_, other.ko_) {
    weights_ = other.weights_;
  }

  FullConnection& FullConnection::operator=(FullConnection other) {
    ki_ = other.ki_;
    ko_ = other.ko_;
    weights_ = other.weights_;
    return *this;
  }

  FullConnection* FullConnection::clone() const { 
    return new FullConnection(*this); 
  }

  std::ostream& FullConnection::put(std::ostream& stream) {
    stream << "C" << weights_;
    return stream;
  }

  void FullConnection::propogate(viennacl::vector<float>& input, viennacl::vector<float>& output) {
    output = viennacl::linalg::prod(weights_, input);
  }

  IdentityConnection::IdentityConnection(unsigned int k) : Connection(k, k) {
    weights_ = viennacl::zero_vector<float>(ki_);
  }

  IdentityConnection::IdentityConnection(const IdentityConnection& other) : Connection(other.ki_, other.ko_) {
    weights_ = other.weights_;
  }

  IdentityConnection& IdentityConnection::operator=(IdentityConnection other) {
    ki_ = other.ki_;
    ko_ = other.ko_;
    weights_ = other.weights_;
    return *this;
  }

  IdentityConnection* IdentityConnection::clone() const {
    return new IdentityConnection(*this);
  }

  std::ostream& IdentityConnection::put(std::ostream& stream) {
    stream << "C" << weights_;
    return stream;
  }

  void IdentityConnection::propogate(viennacl::vector<float>& input, viennacl::vector<float>& output) {
    output = viennacl::linalg::element_prod(weights_, input);
  }

  SparseConnection::SparseConnection(unsigned int k, const viennacl::compressed_matrix<float>& mask) : Connection(k, k) {
    weights_ = mask;
  }

  SparseConnection::SparseConnection(const SparseConnection& other) : Connection(other.ki_, other.ko_) {
    weights_ = other.weights_;
  }

  SparseConnection& SparseConnection::operator=(SparseConnection other) {
    ki_ = other.ki_;
    ko_ = other.ko_;
    weights_ = other.weights_;
    return *this;
  }

  SparseConnection* SparseConnection::clone() const {
    return new SparseConnection(*this);
  }

  std::ostream& SparseConnection::put(std::ostream& stream) {
    stream << "C";
    return stream;
  }

  void SparseConnection::propogate(viennacl::vector<float>& input, viennacl::vector<float>& output) {
    output = viennacl::linalg::prod(weights_, input);
  }

  Connection* new_clone(Connection const& other) {
    return other.clone();
  }

  std::ostream& operator<<(std::ostream & stream, Connection& connection) {
    return connection.put(stream);
  }
}