__kernel void sum(__global float* input, const uint k, __local float* scratch) {
  float sum = 0;
  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    sum += input[i];
  }
  scratch[get_local_id(0)] = sum;
  
  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_local_id(0) == 0) {
    for (uint i = 1; i < get_local_size(0); ++i) {
      sum += scratch[i];
    }
    scratch[0] = sum;
  }
}
