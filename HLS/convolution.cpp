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
      channel_in = stream_in.read ();

      channel_out.data = channel_in.data;
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

#pragma HLS INTERFACE axis port=&stream_in
#pragma HLS INTERFACE axis port=&stream_out

#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

#pragma HLS DATAFLOW
#pragma HLS INLINE region // bring loops in sub-functions to this DATAFLOW region

  static float filter[200000] = {0};
  static int filter_size = 0;

//  StreamChannel channel_in;
//  data_t temp;
//
//  switch (mode)
//  {
//    case 0:
//
//
//      filter_size = output_depth * filter_height * filter_width * input_depth;
//      for (int i = 0; i < filter_size; i++)
//      {
//        channel_in = stream_in.read ();
//        temp.u32 = channel_in.data;
//        filter[i] = temp.f32;
//      }
//      break;
//    case 1:
//      convolution_strm<float> (batches,
//                                input_height, input_width, input_depth,
//                                filter_height, filter_width,
//                                output_height, output_width, output_depth,
//                                stream_in, stream_out,
//                                filter);
//      break;
//    case 2:
//      break;
//    default:;
//  }
  StreamChannel channel_in;
  StreamChannel channel_out;

  for (int i = 0; i < batches; i++)
  {
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

