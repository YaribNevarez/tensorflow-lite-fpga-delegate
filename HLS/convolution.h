#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include <assert.h>
#include <stdint.h>
#include <hls_stream.h>
#include <ap_int.h>
#include "ap_axi_sdata.h"

#define MAX_IMG_ROWS 1080
#define MAX_IMG_COLS 1920

#define TEST_IMG_ROWS 135
#define TEST_IMG_COLS 240
#define TEST_IMG_SIZE (TEST_IMG_ROWS * TEST_IMG_COLS)


#define CHANNEL_WIDTH 64
typedef ap_axis<CHANNEL_WIDTH, 2, 5, 6> StreamChannel;

typedef union
{
  unsigned int u32;
  float f32;
} data_t;

void conv (int batches,
           int input_height,
           int input_width,
           int input_depth,
           int filter_height,
           int filter_width,
           int output_height,
           int output_width,
           int output_depth,
           int mode,
           int * debug,
           hls::stream<StreamChannel> &stream_in,
           hls::stream<StreamChannel> &stream_out);

#endif // CONVOLUTION_H_ not defined

