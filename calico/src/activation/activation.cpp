#include "activation.h"

namespace catnip {
  inline void generic_execute(const std::string& kernel, const viennacl::vector<float>& input, viennacl::vector<float>& output) {
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ActivationProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel(kernel);
    viennacl::ocl::enqueue(k(input, output, cl_uint(input.size()))); 
    viennacl::ocl::get_queue().finish();
  }

  LinearActivator::LinearActivator(float constant) : constant_(constant) {
    ActivationProgram::init();
  }

  void LinearActivator::activate(const viennacl::vector<float>& input, viennacl::vector<float>& output) {
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ActivationProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("linear_activate");
    viennacl::ocl::enqueue(k(input, output, cl_uint(input.size()), constant_));
    viennacl::ocl::get_queue().finish();
  }

  void LinearActivator::derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes) {
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ActivationProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("linear_derivatives");
    viennacl::ocl::enqueue(k(xprimes, cl_uint(xprimes.size()), constant_));
    viennacl::ocl::get_queue().finish();
  }

  SigmoidActivator::SigmoidActivator() {
    ActivationProgram::init();
  }

  void SigmoidActivator::activate(const viennacl::vector<float>& input, viennacl::vector<float>& output) {
    generic_execute("sigmoid_activate", input, output);
  }

  void SigmoidActivator::derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes) {
    generic_execute("sigmoid_derivatives", xs, xprimes);
  }

  TanhActivator::TanhActivator() {
    ActivationProgram::init();
  }

  void TanhActivator::activate(const viennacl::vector<float>& input, viennacl::vector<float>& output) {
    generic_execute("tanh_activate", input, output);
  }

  void TanhActivator::derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes) {
     generic_execute("tanh_derivatives", xs, xprimes);
  }

  SoftmaxActivator::SoftmaxActivator() {
    ActivationProgram::init();
  }

  void SoftmaxActivator::activate(const viennacl::vector<float>& input, viennacl::vector<float>& output) {
    
    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ActivationProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("softmax_activate");
    viennacl::ocl::enqueue(
      k(
        input,
        output,
        cl_uint(input.size())
      )
    );
    viennacl::ocl::get_queue().finish();
  }

  void SoftmaxActivator::derivatives(const viennacl::vector<float>& xs, viennacl::vector<float>& xprimes) {

    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ActivationProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("softmax_derivatives");
    viennacl::ocl::enqueue(
      k(
        xs,
        xprimes,
        cl_uint(xs.size())
      )
    );
    viennacl::ocl::get_queue().finish();
  }
}