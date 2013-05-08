#include "error.h"

namespace catnip {
  void RmsErrorFunction::evaluate(
      const viennacl::vector<float>& predicted, 
      const viennacl::vector<float>& actual,
      viennacl::vector<float>& error) {

    viennacl::ocl::program& prog = viennacl::ocl::current_context().get_program(ErrorProgram::program_name());
    viennacl::ocl::kernel& k = prog.get_kernel("rms");
    //viennacl::ocl::enqueue(
    //    k(
    //      predicted,
    //      actual,
    //      error,
    //      viennacl::ocl::local_mem(sizeof(float) * k.local_work_size())
    //    )
    //);
  }

  void LinearErrorFunction::evaluate(
      const viennacl::vector<float>& predicted, 
      const viennacl::vector<float>& actual,
      viennacl::vector<float>& error) {

    error = actual - predicted;
  }
}