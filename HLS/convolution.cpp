#include "convolution.h"
#include <stdint.h>
#include <algorithm>

#define CONV_PROFILE_NUM_WORDS ((sizeof(ConvProfile)/(DMA_CHANNEL_WIDTH/8)) + (0<(sizeof(ConvProfile)%(DMA_CHANNEL_WIDTH/8))))

#define OUTPUT_BUFFER_SIZE 8



static StreamChannel channel_in;
static StreamChannel channel_out;

template <typename T>
inline T ActivationFunctionWithMinMax(T x, T output_activation_min,
                                      T output_activation_max) {
  using std::max;
  using std::min;
  return min(max(x, output_activation_min), output_activation_max);
}

#define FlatSize(shape)               ((shape)->dims_[0] * (shape)->dims_[1] * (shape)->dims_[2] * (shape)->dims_[3])

#define Offset(shape, i0, i1, i2, i3) ((((i0) * (shape)->dims_[1] + i1) * (shape)->dims_[2] + (i2)) * (shape)->dims_[3] + (i3))


template<typename T>
  static int Convolution_execution (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    ConvProfile & Conv_profile,
                                    float *       Conv_filter,
                                    float *       Conv_bias,
                                    int * debug)
{
  ///////////////////////////////////////////////////////////////////////////////
  static float  StreamPeripheral_inputBuffer[3072] = {0};
  static int    StreamPeripheral_yOffset = 0;
  static int    StreamPeripheral_lookupTable[32] = {0};
  static int    StreamPeripheral_lookupTableRows[32] = {0};
  static int    StreamPeripheral_lookupIndex = 0;
  static int    StreamPeripheral_lookupLength = 0;
  static int    StreamPeripheral_rowCount = 0;
  static int    StreamPeripheral_rowSize = 0;
  ///////////////////////////////////////////////////////////////////////////////
  static int    AXIStream_index = 0;
  static float  AXIStream_outputBuffer[2] = {0};
  static int    AXIStreamOut_index = 0;
  static int    AXIStreamOut_indexLast = 0;
  ///////////////////////////////////////////////////////////////////////////////

  ConvProfile * profile = &Conv_profile;

  TensorShape * input_shape = &profile->input_shape_;
  volatile int batches = input_shape->dims_[0];
  int input_height = input_shape->dims_[1];
  int input_width = input_shape->dims_[2];
  volatile int input_depth = input_shape->dims_[3];

  TensorShape * filter_shape = &profile->filter_shape_;
  volatile int filter_height = filter_shape->dims_[1];
  volatile int filter_width = filter_shape->dims_[2];
  float * filter_data = Conv_filter;

  TensorShape * output_shape = &profile->output_shape_;
  volatile int output_height = output_shape->dims_[1];
  volatile int output_width = output_shape->dims_[2];
  volatile int output_depth = output_shape->dims_[3];

  TensorShape * bias_shape = &profile->bias_shape_;
  float * bias_data = Conv_bias;
  bool bias_data_enable = (bias_shape->dims_[0] == output_depth);

  int stride_height = profile->parameters_.stride_.height_;
  int stride_width = profile->parameters_.stride_.width_;

  int pad_height = profile->parameters_.padding_.height_;
  int pad_width = profile->parameters_.padding_.width_;

  int dilation_height_factor = profile->parameters_.dilation_.height_;
  int dilation_width_factor = profile->parameters_.dilation_.height_;

  float output_activation_max = profile->parameters_.activation_.max_;
  float output_activation_min = profile->parameters_.activation_.min_;

  *debug = 7;

  /*Initialize ()*/
  {
    Data temp_0;
    Data temp_1;

    AXIStream_index = 0;

    AXIStreamOut_index = 0;

    AXIStreamOut_indexLast = FlatSize(output_shape);

    StreamPeripheral_lookupLength = filter_height;

    StreamPeripheral_lookupIndex = StreamPeripheral_lookupLength - 1;

    StreamPeripheral_rowCount = 0;

    StreamPeripheral_rowSize = input_width * input_depth;

    for (int row = 0; row < StreamPeripheral_lookupIndex; row++)
    {
#pragma HLS pipeline
      StreamPeripheral_lookupTable[row] = AXIStream_index;
      StreamPeripheral_lookupTableRows[row] = StreamPeripheral_rowCount++;

      for (int j = 0; j < StreamPeripheral_rowSize; j += 2)
      {
#pragma HLS pipeline
        channel_in = stream_in.read ();
        temp_0.u32 = channel_in.data;
        temp_1.u32 = channel_in.data >> 32;

        StreamPeripheral_inputBuffer[AXIStream_index + j + 0] = temp_0.f32;
        StreamPeripheral_inputBuffer[AXIStream_index + j + 1] = temp_1.f32;
      }
      AXIStream_index += StreamPeripheral_rowSize;
    }

    StreamPeripheral_lookupTable[StreamPeripheral_lookupIndex] = AXIStream_index;
  }

  CONV_OUTPUT_BATCH: for (int batch = 0; batch < batches; ++batch)
  {
#pragma HLS pipeline
    CONV_OUTPUT_ROW: for (int out_y = 0; out_y < output_height; ++out_y)
    {
#pragma HLS pipeline
      const int in_y_origin = (out_y * stride_height) - pad_height;

      /*pushRow()*/
      {
        Data temp_0;
        Data temp_1;

        if (0 <= in_y_origin && StreamPeripheral_rowCount < input_height)
        {
          int i = StreamPeripheral_lookupTable[StreamPeripheral_lookupIndex];

          StreamPeripheral_yOffset = in_y_origin;

          StreamPeripheral_lookupTableRows[StreamPeripheral_lookupIndex] = StreamPeripheral_rowCount++;

          for (int j = 0; j < StreamPeripheral_rowSize; j += 2)
          {
#pragma HLS pipeline
            channel_in = stream_in.read ();
            temp_0.u32 = channel_in.data;
            temp_1.u32 = channel_in.data >> 32;

            StreamPeripheral_inputBuffer[i + j + 0] = temp_0.f32;
            StreamPeripheral_inputBuffer[i + j + 1] = temp_1.f32;
          }
          AXIStream_index += StreamPeripheral_rowSize;

          if (StreamPeripheral_lookupIndex + 1 < StreamPeripheral_lookupLength)
          {
            StreamPeripheral_lookupIndex++;
          }
          else
          {
            StreamPeripheral_lookupIndex = 0;
          }
        }
      }

      CONV_OUTPUT_COL: for (int out_x = 0; out_x < output_width; ++out_x)
      {
#pragma HLS pipeline
        const int in_x_origin = (out_x * stride_width) - pad_width;
        CONV_OUTPUT_CHANNEL: for (int out_channel = 0; out_channel < output_depth; ++out_channel)
        {
#pragma HLS pipeline
          float total = 0.f;
          CONV_FILTER_ROW: for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
#pragma HLS pipeline
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            CONV_FILTER_COL: for (int filter_x = 0; filter_x < filter_width; ++filter_x)
            {
#pragma HLS pipeline
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image = (in_x >= 0)
                  && (in_x < input_width) && (in_y >= 0)
                  && (in_y < input_height);

              if (!is_point_inside_image)
              {
                continue;
              }

              CONV_FILTER_CHANNEL: for (int in_channel = 0; in_channel < input_depth; ++in_channel)
              {
#pragma HLS pipeline
                float input_value;
                /*read()*/
                {
                  int lookupIndex;
                  int i;

                  if (StreamPeripheral_lookupIndex < in_y)
                  {
                    lookupIndex = in_y - StreamPeripheral_yOffset + StreamPeripheral_lookupIndex;
                  }
                  else
                  {
                    lookupIndex = in_y;
                  }

                  if (lookupIndex > StreamPeripheral_lookupLength)
                    lookupIndex -= StreamPeripheral_lookupLength;

                  i = StreamPeripheral_lookupTable[lookupIndex] + input_shape->dims_[3]*in_x + in_channel;

                  input_value = StreamPeripheral_inputBuffer[i];
                }

                float filter_value = filter_data[Offset (filter_shape,
                                                         out_channel, filter_y,
                                                         filter_x, in_channel)];
                total += (input_value * filter_value);
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data_enable)
          {
            bias_value = bias_data[out_channel];
          }

          /*out()*/
          {
            Data temp_0;
            Data temp_1;

            AXIStream_outputBuffer[AXIStreamOut_index % 2] = ActivationFunctionWithMinMax (total + bias_value,
                                                                                           output_activation_min,
                                                                                           output_activation_max);
            if (AXIStreamOut_index % 2 == 2 - 1)
            {
              temp_0.f32 = AXIStream_outputBuffer[0];
              temp_1.f32 = AXIStream_outputBuffer[1];
              channel_out.data =
              ((ap_uint<DMA_CHANNEL_WIDTH>) temp_0.u32) |
              (((ap_uint<DMA_CHANNEL_WIDTH>) temp_1.u32) << 32);
              channel_out.last = (AXIStreamOut_index + 1) == AXIStreamOut_indexLast;
              stream_out.write (channel_out);
            }
            AXIStreamOut_index ++;
          }
        }
      }
    }
  }

  return 0;
}

inline void Convolution_loadProfile (hls::stream<StreamChannel> &stream_in,
                                     hls::stream<StreamChannel> &stream_out,
                                     ConvProfile  & Conv_profile,
                                     int * debug)
{
  Data temp_0;
  Data temp_1;

  *debug = 6;

  // stride_
  channel_in = stream_in.read ();
  Conv_profile.parameters_.stride_.height_   = channel_in.data;
  Conv_profile.parameters_.stride_.width_    = channel_in.data >> 32;

  // dilation_
  channel_in = stream_in.read ();
  Conv_profile.parameters_.dilation_.height_ = channel_in.data;
  Conv_profile.parameters_.dilation_.width_  = channel_in.data >> 32;

  // padding_
  channel_in = stream_in.read ();
  Conv_profile.parameters_.padding_.height_  = channel_in.data;
  Conv_profile.parameters_.padding_.width_   = channel_in.data >> 32;

  // activation_
  channel_in = stream_in.read ();
  temp_0.u32 = channel_in.data;
  temp_1.u32 = channel_in.data >> 32;
  Conv_profile.parameters_.activation_.max_ = temp_0.f32;
  Conv_profile.parameters_.activation_.min_  = temp_1.f32;

  // input_shape_
  channel_in = stream_in.read ();
  Conv_profile.input_shape_.dims_[0] = channel_in.data;
  Conv_profile.input_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  Conv_profile.input_shape_.dims_[2] = channel_in.data;
  Conv_profile.input_shape_.dims_[3] = channel_in.data >> 32;

  // filter_shape_
  channel_in = stream_in.read ();
  Conv_profile.filter_shape_.dims_[0] = channel_in.data;
  Conv_profile.filter_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  Conv_profile.filter_shape_.dims_[2] = channel_in.data;
  Conv_profile.filter_shape_.dims_[3] = channel_in.data >> 32;

  // bias_shape_
  channel_in = stream_in.read ();
  Conv_profile.bias_shape_.dims_[0] = channel_in.data;
  Conv_profile.bias_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  Conv_profile.bias_shape_.dims_[2] = channel_in.data;
  Conv_profile.bias_shape_.dims_[3] = channel_in.data >> 32;

  // output_shape_
  channel_in = stream_in.read ();
  Conv_profile.output_shape_.dims_[0] = channel_in.data;
  Conv_profile.output_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  Conv_profile.output_shape_.dims_[2] = channel_in.data;
  Conv_profile.output_shape_.dims_[3] = channel_in.data >> 32;
}

inline void Convolution_fetchProfile (hls::stream<StreamChannel> &stream_in,
                                     hls::stream<StreamChannel> &stream_out,
                                     ConvProfile & Conv_profile,
                                     int * debug)
{
  Data temp_0;
  Data temp_1;

  *debug = 5;
  // stride_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.parameters_.stride_.height_) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.parameters_.stride_.width_) << 32);
  stream_out.write (channel_out);

  // dilation_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.parameters_.dilation_.height_) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.parameters_.dilation_.width_) << 32);
  stream_out.write (channel_out);

  // padding_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.parameters_.padding_.height_) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.parameters_.padding_.width_) << 32);
  stream_out.write (channel_out);

  // activation_
  temp_0.f32 = Conv_profile.parameters_.activation_.max_;
  temp_1.f32 = Conv_profile.parameters_.activation_.min_;
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) temp_0.u32) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) temp_1.u32) << 32);
  stream_out.write (channel_out);

  // input_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.input_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.input_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.input_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.input_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);

  // filter_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.filter_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.filter_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.filter_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.filter_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);

  // bias_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.bias_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.bias_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.bias_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.bias_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);

  // output_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.output_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.output_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.output_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) Conv_profile.output_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);
}

inline void Convolution_loadFilter (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    ConvProfile & Conv_profile,
                                    float *       Conv_filter,
                                    int * debug)
{
  Data temp_0;
  Data temp_1;
  volatile int filter_size = FlatSize (&Conv_profile.filter_shape_);
  *debug = 4;
  // CONV_LOAD_FILTER_LOOP
  CONV_LOAD_FILTER_LOOP: for (int i = 0; i < filter_size; i += 2)
  {
#pragma HLS pipeline
    channel_in = stream_in.read ();
    temp_0.u32 = channel_in.data;
    Conv_filter[i] = temp_0.f32;
    temp_1.u32 = channel_in.data >> 32;
    Conv_filter[i + 1] = temp_1.f32;
  }
}

inline void Convolution_loadBias (hls::stream<StreamChannel> &stream_in,
                                  hls::stream<StreamChannel> &stream_out,
                                  ConvProfile & Conv_profile,
                                  float *       Conv_bias,
                                  int * debug)
{
  Data temp_0;
  Data temp_1;
  volatile int bias_size = FlatSize (&Conv_profile.bias_shape_);
  *debug = 3;
  // CONV_LOAD_BIAS_LOOP
  CONV_LOAD_BIAS_LOOP: for (int i = 0; i < bias_size; i += 2)
  {
#pragma HLS pipeline
    channel_in = stream_in.read ();
    temp_0.u32 = channel_in.data;
    Conv_bias[i] = temp_0.f32;
    temp_1.u32 = channel_in.data >> 32;
    Conv_bias[i + 1] = temp_1.f32;
  }
}

inline void Convolution_fetchFilter (hls::stream<StreamChannel> &stream_in,
                                     hls::stream<StreamChannel> &stream_out,
                                     ConvProfile & Conv_profile,
                                     float *       Conv_filter,
                                     int * debug)
{
  Data temp_2;
  Data temp_3;
  volatile int filter_size = FlatSize (&Conv_profile.filter_shape_);
  *debug = 2;
  // CONV_LOAD_FILTER_LOOP
  CONV_FETCH_FILTER_LOOP: for (int i = 0; i < filter_size; i += 2)
  {
//#pragma HLS pipeline
    temp_2.f32 = Conv_filter[i];
    temp_3.f32 = Conv_filter[i + 1];

    channel_out.data = ((((ap_uint<64> ) temp_3.u32) << 32) & 0xFFFFFFFF00000000) | ((ap_uint<64> ) temp_2.u32);
    //channel_out.last = filter_size <= i + 2;
    stream_out.write (channel_out);
  }
}

inline void Convolution_fetchBias (hls::stream<StreamChannel> &stream_in,
                                   hls::stream<StreamChannel> &stream_out,
                                   ConvProfile & Conv_profile,
                                   float *       Conv_bias,
                                   int * debug)
{
  Data temp_2;
  Data temp_3;
  volatile int bias_size = FlatSize (&Conv_profile.bias_shape_);
  // CONV_LOAD_BIAS_LOOP
  *debug = 1;
  CONV_FETCH_BIAS_LOOP: for (int i = 0; i < bias_size; i += 2)
  {
//#pragma HLS pipeline
    temp_2.f32 = Conv_bias[i];
    temp_3.f32 = Conv_bias[i + 1];

    channel_out.data = ((((ap_uint<64> ) temp_3.u32) << 32) & 0xFFFFFFFF00000000) | ((ap_uint<64> ) temp_2.u32);
    //channel_out.last = bias_size <= i + 2;
    stream_out.write (channel_out);
  }
}

int conv (ConvExecutionMode mode,
          hls::stream<StreamChannel> &stream_in,
          hls::stream<StreamChannel> &stream_out,
          int * debug)
{
#pragma HLS INTERFACE s_axilite port=mode  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=debug bundle=CRTL_BUS

#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out

#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

  ///////////////////////////////////////////////////////////////////////////////
  #define CONV_FILTER_BUFFER_SIZE 128000
  #define CONV_BIAS_BUFFER_SIZE   512

  static ConvProfile  Conv_profile;
  static float        Conv_filter[CONV_FILTER_BUFFER_SIZE];
  static float        Conv_bias[CONV_BIAS_BUFFER_SIZE];
  ///////////////////////////////////////////////////////////////////////////////

//#pragma HLS DATAFLOW
//#pragma HLS INLINE region // bring loops in sub-functions to this DATAFLOW region

  int rc = 0;
  channel_out.keep = -1;
  channel_out.strb = -1;

  switch (mode)
  {
    case CONV_LOAD_PROFILE_PACKAGE:
      Convolution_loadProfile  (stream_in, stream_out, Conv_profile, debug);
      Convolution_loadFilter   (stream_in, stream_out, Conv_profile, Conv_filter, debug);
      Convolution_loadBias     (stream_in, stream_out, Conv_profile, Conv_bias, debug);
      break;
//    case CONV_LOAD_PROFILE:
//      Convolution_loadProfile  (stream_in, stream_out, debug);
//      break;
    case CONV_FETCH_PROFILE:
      Convolution_fetchProfile (stream_in, stream_out, Conv_profile, debug);
      break;
//    case CONV_LOAD_FILTER:
//      Convolution_loadFilter   (stream_in, stream_out, debug);
//      break;
//    case CONV_FETCH_FILTER:
//      Convolution_fetchFilter  (stream_in, stream_out, debug);
//      break;
//    case CONV_LOAD_BIAS:
//      Convolution_loadBias     (stream_in, stream_out, debug);
//      break;
//    case CONV_FETCH_BIAS:
//      Convolution_fetchBias    (stream_in, stream_out, debug);
//      break;
    case CONV_EXECUTION:
      Convolution_execution<float> (stream_in, stream_out, Conv_profile, Conv_filter, Conv_bias, debug);
      break;
    default:
      rc = -1;
  }

  channel_out.last = 0;

  return rc;
}

