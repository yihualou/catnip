__kernel void convolve2d_full(
    const __global float* in,
    __global float* out,
    __global float* filter,
    __local float* scratch,
    uint in_width,
    uint in_height,
    uint filter_width,
    uint filter_height) {

  const uint half_filter_width = filter_width / 2;
  const uint half_filter_height = filter_height / 2;

  const uint in_block_orig_width = in_width / get_num_groups(0);
  const uint in_block_orig_height = in_height / get_num_groups(1);

  uint x_start, x_end, y_start, y_end;

  // Check if we have more work items than pixels.
  if (in_block_orig_width == 0 || in_block_orig_height == 0) {
    uint gx = get_group_id(0), gy = get_group_id(1);

    // Out of bounds?
    if (gx >= in_width || gy >= in_height) {
      return;
    }
    x_start = gx; 
    x_end = gx + filter_width;
    y_start = gy;
    y_end = gy + filter_height;

  } else {
    x_start = get_group_id(0) * in_block_orig_width;
    x_end = x_start + in_block_orig_width;
    y_start = get_group_id(1) * in_block_orig_height;
    y_end = y_start + in_block_orig_height;

    // Overlap on the left?
	  if (x_start != 0) {
      x_start -= half_filter_width; 
    } 
    // Overlap on the right?
    else if (x_end < in_width) {
      x_end += half_filter_width;
    }
  
    // Overlap on the top?
    if (y_start != 0) {
      y_start -= half_filter_height; 
    } 
    // Overlap on the bottom?
    else if (y_end < in_height) {
      y_end += half_filter_height;
    }

    // Clamp end points.
    x_end = max(x_end, in_width);
    y_end = max(y_end, in_height);
  }

  // Add padding to borders.
  uint lborder = x_start == 0 ? filter_width - 1 : 0; 
  uint rborder = x_end == in_width - 1 ? filter_width - 1 : 0;
  uint tborder = y_start == 0 ? filter_height - 1: 0;
  uint bborder = y_end == in_height - 1 ? filter_height - 1: 0;

  // This is the in block width AFTER overlaps and paddding have been applied.
  uint in_block_width = (x_end - x_start) + lborder + rborder;

  // Copy data into local cache.
  for (uint y = y_start; y < y_end; ++y) {
    uint local_y = (y - y_start) + tborder;
    for (uint x = x_start; x < x_end; ++x) {
      uint local_x = (x - x_start) + lborder;
      scratch[local_y * in_block_width + (x - x_start)] = in[y * in_width + x];
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint out_width = in_width + filter_width - 1;

  const uint out_block_width = in_block_width + filter_width - 1;
  const uint out_block_height = in_block_width + filter_height - 1;

  // Convolve.
  for (uint y = 0; y < out_block_height; ++y) {
    uint out_y = y + y_start;
    for (uint x = 0; x < out_block_width; ++x) {

      float sum = 0.0f;
      for (uint ky = 0; ky < filter_height; ++ky) {
        for (uint kx = 0; kx < filter_width; ++kx) {
          uint filter_idx = ky * filter_width + kx;
          sum += scratch[(y + ky) * (x_end - x_start) + (x + kx)] * filter[filter_idx]; 
        }
      }

      // Store data.
      out[out_y * out_width + (x + x_start)] = sum;
    }
  }
}

// Follows the format of MATLAB's "valid" conv setting. If a m x n float input array is
// convolved with a a x b size filter, the final image will be (m - a + 1) x (n - b + 1).
// MUST be called with a 2D work group configuration.
__kernel void convolve2d_valid(
    const __global float* in,
    __global float* out,
    __global float* filter,
    __local float* scratch,
    uint in_width,
    uint in_height,
    uint filter_width,
    uint filter_height) {

  const uint half_filter_width = filter_width / 2;
  const uint half_filter_height = filter_height / 2;

  const uint in_block_orig_width = in_width / get_num_groups(0);
  const uint in_block_orig_height = in_height / get_num_groups(1);

  uint x_start, x_end, y_start, y_end;

  // Check if we have more work items than pixels.
  if (in_block_orig_width == 0 || in_block_orig_height == 0) {
    uint gx = get_group_id(0), gy = get_group_id(1);

    // Out of bounds?
    if (gx >= in_width || gy >= in_height) {
      return;
    }
    x_start = gx; 
    x_end = gx + filter_width;
    y_start = gy;
    y_end = gy + filter_height;

  } else {
    x_start = get_group_id(0) * in_block_orig_width;
    x_end = x_start + in_block_orig_width;
    y_start = get_group_id(1) * in_block_orig_height;
    y_end = y_start + in_block_orig_height;

    // Overlap on the left?
	  if (x_start != 0) {
      x_start -= half_filter_width; 
    } 
    // Overlap on the right?
    else if (x_end < in_width) {
      x_end += half_filter_width;
    }
  
    // Overlap on the top?
    if (y_start != 0) {
      y_start -= half_filter_height; 
    } 
    // Overlap on the bottom?
    else if (y_end < in_height) {
      y_end += half_filter_height;
    }

    // Clamp end points.
    x_end = max(x_end, in_width);
    y_end = max(y_end, in_height);
  }

  // This is the in block width AFTER overlaps have been applied.
  uint in_block_width = (x_end - x_start);

  // Copy data into local cache.
  for (uint y = y_start; y < y_end; ++y) {
    uint local_y = y - y_start;
    for (uint x = x_start; x < x_end; ++x) {
      scratch[local_y * in_block_width + (x - x_start)] = in[y * in_width + x];
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint out_width = in_width - filter_width + 1;

  const uint out_block_width = in_block_width - filter_width + 1;
  const uint out_block_height = in_block_width - filter_height + 1;

  // Convolve.
  for (uint y = 0; y < out_block_height; ++y) {
    uint out_y = y + y_start;
    for (uint x = 0; x < out_block_width; ++x) {

      float sum = 0.0f;
      for (uint ky = 0; ky < filter_height; ++ky) {
        for (uint kx = 0; kx < filter_width; ++kx) {
          uint filter_idx = ky * filter_width + kx;
          sum += scratch[(y + ky) * (x_end - x_start) + (x + kx)] * filter[filter_idx]; 
        }
      }

      // Store data.
      out[out_y * out_width + (x + x_start)] = sum;
    }
  }
}

__kernel void upsample2d(
    const __global float* in,
    __global float* out,
    uint in_width,
    uint in_height,
    uint factor) {

  uint out_width = in_width * factor;
  uint out_height = in_height * factor;
  for (uint y = get_global_id(1); y < in_height; y += get_global_size(1)) {
    for (uint x = get_global_id(0); x < in_width; x += get_global_size(0)) {

      float val = in[y * in_width + x];
      for (uint ox = x * factor; ox < (x + 1) * factor; ox++) {
        for (uint oy = y * factor; oy < (y + 1) * factor; oy++) {
          out[oy * out_width + ox] = val;
        }
      }

    }
  }
}

__kernel void maxpool2d(
    const __global float* in,
    __global float* out,
    uint in_width,
    uint in_height,
    uint factor) {

  uint out_width = in_width / factor;
  uint out_height = in_height / factor;
  for (uint y = get_global_id(1); y < out_height; y += get_global_size(1)) {
    for (uint x = get_global_id(0); x < out_width; x += get_global_size(0)) {

      for (uint ix = x * factor; ix < (x + 1) * factor; ix++) {
        for (uint iy = y * factor; iy < (y + 1) * factor; iy++) {
          out[y * out_width + x] = max(out[y * out_width + x], in[iy * in_width + ix]);
        }
      }

    }
  }
}