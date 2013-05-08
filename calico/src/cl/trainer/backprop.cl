__kernel void backprop_update_local(
    __global const unsigned int* row_indices,
    __global const unsigned int* column_indices, 
    __global float* elements,
    __global const float* input,
    __global const float* error, 
    __global const float* derivative,
    const uint k,
    const float alpha) { 

  for (uint row = get_global_id(0); row < k; row += get_global_size(0)) {
    uint row_end = row_indices[row + 1];
    for (uint i = row_indices[row]; i < row_end; ++i) {
      uint col = column_indices[i];
      elements[i] += alpha * error[row] * derivative[row] * input[col]; 
    }
  }
}