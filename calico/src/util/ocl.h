#ifndef OCL_H
#define OCL_H

#include <viennacl/ocl/backend.hpp>
#include <boost/unordered_set.hpp>
#include <string>

namespace catnip {
  template<class Derived>
  struct OclProgram {
    static void init() {
      std::string program_name = Derived::program_name();
      std::string filename = OCL_ROOT_DIR + Derived::relative_path();

      unsigned int num_kernels;
      std::string* kernel_names = Derived::kernel_names(num_kernels);

      viennacl::ocl::context & context = viennacl::ocl::current_context();
      std::map<cl_context, bool>& init_done = Derived::init_done;
      if (!init_done[context.handle().get()]) {
        std::string source = get_file_contents(filename.c_str());

        context.build_options("-I"OCL_ROOT_DIR);
        context.add_program(source, program_name);

        viennacl::ocl::program & program = context.get_program(program_name);

        for (unsigned int i = 0; i < num_kernels; ++i) {
          program.add_kernel(kernel_names[i]);
        }

        init_done[context.handle().get()] = true;
      }
    }
  };

  extern std::string get_file_contents(const char *filename);
}

#endif