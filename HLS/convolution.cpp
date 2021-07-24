#include "convolution.h"

template<typename T>
  static void convolution_strm (int batches,
                                int input_height, int input_width, int input_depth,
                                int filter_height, int filter_width,
                                int output_height, int output_width, int output_depth,
                                hls::stream<StreamChannel> &stream_in, hls::stream<StreamChannel> &stream_out,
                                T *filter)
  {
    StreamChannel channel_in;
    StreamChannel channel_out;

    for (int i = 0; i < batches; i++)
    {
#pragma HLS pipeline
      channel_in = stream_in.read ();

      channel_out.data = channel_in.data * 2;
      channel_out.keep = channel_in.keep;
      channel_out.strb = channel_in.strb;
      channel_out.user = channel_in.user;
      channel_out.last = channel_in.last;
      channel_out.id = channel_in.id;
      channel_out.dest = channel_in.dest;

      stream_out.write (channel_out);
    }
  }


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
           hls::stream<StreamChannel> &stream_out)
{
#pragma HLS INTERFACE s_axilite port=batches bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=input_height bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=input_width bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=input_depth bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=filter_height bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=filter_width bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=output_height bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=output_width bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=output_depth bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=mode bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=debug bundle=CRTL_BUS

#pragma HLS INTERFACE axis port=&stream_in
#pragma HLS INTERFACE axis port=&stream_out

#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

#pragma HLS DATAFLOW
#pragma HLS INLINE region // bring loops in sub-functions to this DATAFLOW region

  static float filter[120000] = {0};
  static int filter_size = 0;

  StreamChannel channel_in;
  StreamChannel channel_out;
  data_t temp_0;
  data_t temp_1;
  data_t temp_2;
  data_t temp_3;

  channel_out.keep = -1;
  channel_out.strb = -1;

  filter_size = output_depth * filter_height * filter_width * input_depth;

  switch (mode)
  {
    case 0:
      FILTER_LOAD: for (int i = 0; i < filter_size; i += 2)
      {
#pragma HLS pipeline
        channel_in = stream_in.read ();
        temp_0.u32 = channel_in.data;
        filter[i] = temp_0.f32;
        temp_1.u32 = channel_in.data >> 32;
        filter[i + 1] = temp_1.f32;
      }
      break;
    case 1:
      FILTER_FETCH: for (int i = 0; i < filter_size; i += 2)
      {
#pragma HLS pipeline
        temp_2.f32 = filter[i];
        temp_3.f32 = filter[i + 1];

        channel_out.data = ((((ap_uint<64> ) temp_3.u32) << 32) & 0xFFFFFFFF00000000) | ((ap_uint<64> ) temp_2.u32);
        channel_out.last = filter_size <= i + 2;
        stream_out.write (channel_out);
      }
      break;
    case 2:
      convolution_strm<float> (batches,
                                input_height, input_width, input_depth,
                                filter_height, filter_width,
                                output_height, output_width, output_depth,
                                stream_in, stream_out,
                                filter);
      break;
    case 3:
      break;
    default:;
  }
}

