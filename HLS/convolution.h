#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include <assert.h>
#include <stdint.h>
#include <hls_stream.h>
#include <ap_int.h>
#include "ap_axi_sdata.h"
#include "conv_hls.h"

#define CONV_FILTER_BUFFER_SIZE     256*1024
#define CONV_BIAS_BUFFER_SIZE       256
#define CONV_INPUT_BUFFER_SIZE      4*1024

#define FIXED_POINT                 false

#if !FIXED_POINT
#define HYBRID_LOGARITHMIC          true
#if HYBRID_LOGARITHMIC
#define CUSTOM_SIGN_BIT             1
#define CUSTOM_EXPONENT_BIT_WIDTH   5
#define CUSTOM_MANTISSA_BIT_WIDTH   1


#define CORRECTION                  true


// Set to 0 for normalized numbers [0 - 1), All exponents are on the left side, so they are stored without sign
#define CUSTOM_EXPONENT_SIGN_BIT    1
#endif
#endif

typedef ap_axis<DMA_CHANNEL_WIDTH, 2, 5, 6> StreamChannel;

int conv (ConvExecutionMode mode,
          hls::stream<StreamChannel> &stream_in,
          hls::stream<StreamChannel> &stream_out,
          int * debug);

#endif // CONVOLUTION_H_ not defined

