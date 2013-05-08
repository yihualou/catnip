#include "rng/mwc64x.cl"

__kernel void rbm_activate(
    const __global float* in,
    __global float* out,
    const uint k,
    const uint base_offset) {

  mwc64x_state_t rng;
  ulong samples = k / get_global_size(0);
  MWC64X_SeedStreams(&rng, base_offset, 2 * samples);

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    uint rand = MWC64X_NextUint(&rng);
    uint prob = UINT_MAX / (1.0f + exp(-in[i]));
    out[i] = rand <= prob ? 1 : 0;
  }  
}