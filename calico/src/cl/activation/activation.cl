__kernel void linear_activate(
    const __global float* in,
    __global float* out,
    const uint k,
    const float c) {

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    out[i] = c * in[i];
  }
}

__kernel void linear_derivatives(
    __global float* out,
    const uint k,
    const float c) {

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    out[i] = c;
  }
}

__kernel void sigmoid_activate(
    const __global float* in,
    __global float* out,
    const uint k) {

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    out[i] = 1.0f / (1.0f + exp(-in[i]));
  }
}

__kernel void sigmoid_derivatives(
    const __global float* in,
    __global float* out,
    const uint k) {

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    float s = 1.0f / (1.0f + exp(-in[i]));
    out[i] = s * (1 - s);
  }
}

__kernel void tanh_activate(
    const __global float* in,
    __global float* out,
    const uint k) {

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    out[i] = tanh(in[i]);
  }
}

__kernel void tanh_derivatives(
    const __global float* in,
    __global float* out,
    const uint k) {

  for (uint i = get_global_id(0); i < k; i += get_global_size(0)) {
    float t = tanh(in[i]);
    out[i] = 1 - t * t;
  }
}

__kernel void softmax_activate(
    const __global float* in,
    __global float* out,
    const uint k) {

  if (get_global_id(0) == 0) {
    float m = -INFINITY;
    for (uint i = 0; i < k; ++i) {
      if (in[i] > m) {
        m = in[i];
      }
    }

    float sum = 0.0f;
    for (uint i = 0; i < k; ++i) {
      float val = exp(in[i] - m);
      out[i] = val;
      sum += val;
    }

    for (uint i = 0; i < k; ++i) {
      out[i] /= sum;
    }
  }
}

__kernel void softmax_derivatives(
    const __global float* in,
    __global float* out,
    const uint k) {

  if (get_global_id(0) == 0) {
    float m = -INFINITY;
    for (uint i = 0; i < k; ++i) {
      if (in[i] > m) {
        m = in[i];
      }
    }

    float sum = 0.0f;
    for (uint i = 0; i < k; ++i) {
      float val = exp(in[i] - m);
      out[i] = val;
      sum += val;
    }

    for (uint i = 0; i < k; ++i) {
      float val = out[i] /= sum;
      out[i] = val * (1 - val);
    }
  }
}
