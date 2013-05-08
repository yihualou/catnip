#ifndef CIFAR_H_
#define CIFAR_H_

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "global_defines.h"

#include "labeled.h"

namespace catnip {
  class CifarDataset : public LabeledDataset {
    public:
      struct Image {
         char label;
         char red[1024];
         char green[1024];
         char blue[1024];
      };

      CifarDataset(int input_size, int output_size) : LabeledDataset(input_size, output_size) {}

      void load(const std::string& filename, int size);
  };
}

#endif // CIFAR_H_
