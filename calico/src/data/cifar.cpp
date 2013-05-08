#include "cifar.h"

namespace catnip {
  void CifarDataset::load(const std::string& filename, int size) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in) {
      boost::array<float, 3 * 1024> input;
      boost::array<float, 10> output;

      for (int i = 0; i < size; ++i) {
        Image img;
        in.read(reinterpret_cast<char*>(&img), sizeof(Image));

        for (int i = 0; i < 3 * 1024; ++i) {
          input[i] = img.red[i];
        }
        
        for (int i = 0; i < 10; ++i) {
          output[i] = 0.0f;
        }
        output[(int) img.label] = 1.0f;

        add_observation(input, output);
      }
      return;
    }
    throw(errno);
  }
}