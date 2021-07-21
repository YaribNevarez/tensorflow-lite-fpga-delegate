#include <iostream>
#include <cstdlib>

#include "convolution.h"

using namespace std;

int main (void)
{
  data_t * const src_img = new data_t[TEST_IMG_ROWS * TEST_IMG_COLS];
  data_t * const ref_img = new data_t[TEST_IMG_ROWS * TEST_IMG_COLS];
  int batches = 1;
  int input_height = 32;
  int input_width = 32;
  int input_depth = 3;
  int filter_height = 3;
  int filter_width = 3;
  int output_height = 32;
  int output_width = 32;
  int output_depth = 32;
  int mode = 0;
  hls::stream<StreamChannel> stream_in ("stream_in");
  hls::stream<StreamChannel> stream_out ("stream_out");

  int err_cnt = 0;
  int ret_val = 20;

  conv (batches,
        input_height,
        input_width,
        input_depth,
        filter_height,
        filter_width,
        output_height,
        output_width,
        output_depth,
        mode,
        stream_in,
        stream_out);

  delete[] src_img;
  delete[] ref_img;

  return ret_val;
}

