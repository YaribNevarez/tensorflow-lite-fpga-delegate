#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include <assert.h>
#include <stdint.h>
#include <hls_stream.h>
#include <ap_int.h>
#include "ap_axi_sdata.h"
#include "conv_def.h"

typedef ap_axis<DMA_CHANNEL_WIDTH, 2, 5, 6> StreamChannel;

int conv (ConvExecutionMode mode,
          hls::stream<StreamChannel> &stream_in,
          hls::stream<StreamChannel> &stream_out,
          int * debug);

#endif // CONVOLUTION_H_ not defined

