#include "rng/mwc64x.cl"
#include "util/sum.cl"

__kernel void random_initialize(
    __global float* mat,
    const uint k,
    const float magnitude,
    const uint base_offset) {

  mwc64x_state_t rng;
  ulong samples = max((ulong) k / get_global_size(0), 1UL);
  MWC64X_SeedStreams(&rng, base_offset, 2 * samples);

  // Credit goes to http://xor0110.wordpress.com/2010/09/24/how-to-generate-floating-point-random-numbers-efficiently/
  union {
    unsigned int i;
    float f;
  } frand;
 
  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    frand.i = MWC64X_NextUint(&rng) & 0x007fffff | 0x40000000;
    mat[i] = (frand.f - 3.0) * magnitude;
  }  
}

__kernel void fan_in_initialize(
    __global float* mat,
    const uint k,
    const float magnitude,
    const float fan_in,
    const uint base_offset) {
      
  random_initialize(mat, k, 1.0f, base_offset);

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    mat[i] /= fan_in; 
  }
}
