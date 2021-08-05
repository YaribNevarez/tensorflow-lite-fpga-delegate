#include "convolution.h"
#include <stdint.h>
#include <algorithm>

#define CONV_PROFILE_NUM_WORDS ((sizeof(ConvProfile)/(DMA_CHANNEL_WIDTH/8)) + (0<(sizeof(ConvProfile)%(DMA_CHANNEL_WIDTH/8))))

#define CONV_FILTER_BUFFER_SIZE 128000
#define CONV_BIAS_BUFFER_SIZE   512
#define CONV_OUTPUT_BUFFER_SIZE 8

static ConvProfile profile_instance_;
static float filter_instance_[CONV_FILTER_BUFFER_SIZE];
static float bias_instance_[CONV_BIAS_BUFFER_SIZE];

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

///////////////////////////////////////////////////////////////////////////////

static float StreamPeripheral_inputBuffer[3072] = {0};
static int StreamPeripheral_yOffset = 0;
static int StreamPeripheral_lookupTable[32] = {0};
static int StreamPeripheral_lookupTableRows[32] = {0};
static int StreamPeripheral_lookupIndex = 0;
static int StreamPeripheral_lookupLength = 0;
static int StreamPeripheral_rowCount = 0;

static int AXIStream_index = 0;
static float AXIStream_inputBuffer[2] = {0};
static float AXIStream_outputBuffer[2] = {0};
static int AXIStreamOut_index = 0;
static int AXIStreamOut_indexLast = 0;


float StreamPeripheral_inputData (TensorShape input_shape,
                                  int batch,
                                  int in_y,
                                  int in_x,
                                  int in_channel)
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

  i = StreamPeripheral_lookupTable[lookupIndex] + input_shape.dims_[3]*in_x + in_channel;

  return StreamPeripheral_inputBuffer[i];
}

void StreamPeripheral_outputData (hls::stream<StreamChannel> &stream_out, float output)
{
  Data temp_0;
  Data temp_1;

  AXIStream_outputBuffer[AXIStreamOut_index % 2] = output;
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

void StreamPeripheral_initialize (hls::stream<StreamChannel> &stream_in,
                                 int input_depth,
                                 int input_width,
                                 int filter_height,
                                 int output_length)
{
  Data temp_0;
  Data temp_1;

  AXIStream_index = 0;

  AXIStreamOut_index = 0;

  AXIStreamOut_indexLast = output_length;

  StreamPeripheral_lookupLength = filter_height;

  StreamPeripheral_lookupIndex = StreamPeripheral_lookupLength - 1;

  StreamPeripheral_rowCount = 0;

  for (int row = 0; row < StreamPeripheral_lookupIndex; row++)
  {
    StreamPeripheral_lookupTable[row] = AXIStream_index;
    StreamPeripheral_lookupTableRows[row] = StreamPeripheral_rowCount++;
    for (int col = 0; col < input_width; col++)
    {
      for (int chan = 0; chan < input_depth; chan++)
      {
        if (AXIStream_index%2 == 0)
        {
          channel_in = stream_in.read ();
          temp_0.u32 = channel_in.data;
          temp_1.u32 = channel_in.data >> 32;

          AXIStream_inputBuffer[0] = temp_0.f32;
          AXIStream_inputBuffer[1] = temp_1.f32;
        }
        StreamPeripheral_inputBuffer[AXIStream_index] = AXIStream_inputBuffer[AXIStream_index%2];
        AXIStream_index++;
      }
    }
  }

  StreamPeripheral_lookupTable[StreamPeripheral_lookupIndex] = AXIStream_index;
}

void StreamPeripheral_pushSlice (hls::stream<StreamChannel> &stream_in,
                                  int input_depth,
                                  int input_width,
                                  int input_height,
                                  int filter_height,
                                  int in_y_origin)
{
  Data temp_0;
  Data temp_1;

  if (0 <= in_y_origin && StreamPeripheral_rowCount < input_height)
  {
    int i = StreamPeripheral_lookupTable[StreamPeripheral_lookupIndex];

    StreamPeripheral_yOffset = in_y_origin;

    StreamPeripheral_lookupTableRows[StreamPeripheral_lookupIndex] = StreamPeripheral_rowCount++;

    for (int col = 0; col < input_width; col++)
    {
      for (int chan = 0; chan < input_depth; chan++)
      {
        if (AXIStream_index%2 == 0)
        {
          channel_in = stream_in.read ();
          temp_0.u32 = channel_in.data;
          temp_1.u32 = channel_in.data >> 32;

          AXIStream_inputBuffer[0] = temp_0.f32;
          AXIStream_inputBuffer[1] = temp_1.f32;
        }

        StreamPeripheral_inputBuffer[i++] = AXIStream_inputBuffer[AXIStream_index%2];
        AXIStream_index++;
      }
    }

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
///////////////////////////////////////////////////////////////////////////////

template<typename T>
  static int Convolution_execution (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    int * debug)
{
  ConvProfile * profile = &profile_instance_;

  TensorShape * input_shape = &profile->input_shape_;
  volatile int batches = input_shape->dims_[0];
  int input_height = input_shape->dims_[1];
  int input_width = input_shape->dims_[2];
  volatile int input_depth = input_shape->dims_[3];

  TensorShape * filter_shape = &profile->filter_shape_;
  volatile int filter_height = filter_shape->dims_[1];
  volatile int filter_width = filter_shape->dims_[2];
  float * filter_data = filter_instance_;

  TensorShape * output_shape = &profile->output_shape_;
  volatile int output_height = output_shape->dims_[1];
  volatile int output_width = output_shape->dims_[2];
  volatile int output_depth = output_shape->dims_[3];

  TensorShape * bias_shape = &profile->bias_shape_;
  float * bias_data = bias_instance_;
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

  StreamPeripheral_initialize (stream_in,
                                input_depth,
                                input_width,
                                filter_height,
                                FlatSize(output_shape));

  CONV_OUTPUT_BATCH: for (int batch = 0; batch < batches; ++batch)
  {
    CONV_OUTPUT_ROW: for (int out_y = 0; out_y < output_height; ++out_y)
    {
      const int in_y_origin = (out_y * stride_height) - pad_height;

      StreamPeripheral_pushSlice (stream_in,
                                  input_depth,
                                  input_width,
                                  input_height,
                                  filter_height,
                                  in_y_origin);

      CONV_OUTPUT_COL: for (int out_x = 0; out_x < output_width; ++out_x)
      {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        CONV_OUTPUT_CHANNEL: for (int out_channel = 0; out_channel < output_depth; ++out_channel)
        {
          float total = 0.f;
          CONV_FILTER_ROW: for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            CONV_FILTER_COL: for (int filter_x = 0; filter_x < filter_width; ++filter_x)
            {
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
                //float input_value = input_data[Offset (input_shape, batch, in_y,
                //                                       in_x, in_channel)];
                float input_value = StreamPeripheral_inputData (*input_shape, batch, in_y,
                                                      in_x, in_channel);

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
          //output_data[Offset (output_shape, batch, out_y, out_x, out_channel)] =
          StreamPeripheral_outputData(stream_out,
                                      ActivationFunctionWithMinMax (total + bias_value,
                                            output_activation_min,
                                            output_activation_max));
        }
      }
    }
  }

  return 0;
}

inline void Convolution_loadProfile (hls::stream<StreamChannel> &stream_in,
                                     hls::stream<StreamChannel> &stream_out,
                                     int * debug)
{
  Data temp_0;
  Data temp_1;

  *debug = 6;

  // stride_
  channel_in = stream_in.read ();
  profile_instance_.parameters_.stride_.height_   = channel_in.data;
  profile_instance_.parameters_.stride_.width_    = channel_in.data >> 32;

  // dilation_
  channel_in = stream_in.read ();
  profile_instance_.parameters_.dilation_.height_ = channel_in.data;
  profile_instance_.parameters_.dilation_.width_  = channel_in.data >> 32;

  // padding_
  channel_in = stream_in.read ();
  profile_instance_.parameters_.padding_.height_  = channel_in.data;
  profile_instance_.parameters_.padding_.width_   = channel_in.data >> 32;

  // activation_
  channel_in = stream_in.read ();
  temp_0.u32 = channel_in.data;
  temp_1.u32 = channel_in.data >> 32;
  profile_instance_.parameters_.activation_.max_ = temp_0.f32;
  profile_instance_.parameters_.activation_.min_  = temp_1.f32;

  // input_shape_
  channel_in = stream_in.read ();
  profile_instance_.input_shape_.dims_[0] = channel_in.data;
  profile_instance_.input_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  profile_instance_.input_shape_.dims_[2] = channel_in.data;
  profile_instance_.input_shape_.dims_[3] = channel_in.data >> 32;

  // filter_shape_
  channel_in = stream_in.read ();
  profile_instance_.filter_shape_.dims_[0] = channel_in.data;
  profile_instance_.filter_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  profile_instance_.filter_shape_.dims_[2] = channel_in.data;
  profile_instance_.filter_shape_.dims_[3] = channel_in.data >> 32;

  // bias_shape_
  channel_in = stream_in.read ();
  profile_instance_.bias_shape_.dims_[0] = channel_in.data;
  profile_instance_.bias_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  profile_instance_.bias_shape_.dims_[2] = channel_in.data;
  profile_instance_.bias_shape_.dims_[3] = channel_in.data >> 32;

  // output_shape_
  channel_in = stream_in.read ();
  profile_instance_.output_shape_.dims_[0] = channel_in.data;
  profile_instance_.output_shape_.dims_[1] = channel_in.data >> 32;
  channel_in = stream_in.read ();
  profile_instance_.output_shape_.dims_[2] = channel_in.data;
  profile_instance_.output_shape_.dims_[3] = channel_in.data >> 32;
}

inline void Convolution_fetchProfile (hls::stream<StreamChannel> &stream_in,
                                     hls::stream<StreamChannel> &stream_out,
                                     int * debug)
{
  Data temp_0;
  Data temp_1;

  *debug = 5;
  // stride_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.parameters_.stride_.height_) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.parameters_.stride_.width_) << 32);
  stream_out.write (channel_out);

  // dilation_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.parameters_.dilation_.height_) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.parameters_.dilation_.width_) << 32);
  stream_out.write (channel_out);

  // padding_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.parameters_.padding_.height_) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.parameters_.padding_.width_) << 32);
  stream_out.write (channel_out);

  // activation_
  temp_0.f32 = profile_instance_.parameters_.activation_.max_;
  temp_1.f32 = profile_instance_.parameters_.activation_.min_;
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) temp_0.u32) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) temp_1.u32) << 32);
  stream_out.write (channel_out);

  // input_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.input_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.input_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.input_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.input_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);

  // filter_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.filter_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.filter_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.filter_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.filter_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);

  // bias_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.bias_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.bias_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.bias_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.bias_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);

  // output_shape_
  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.output_shape_.dims_[0]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.output_shape_.dims_[1]) << 32);
  stream_out.write (channel_out);

  channel_out.data =
  ((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.output_shape_.dims_[2]) |
  (((ap_uint<DMA_CHANNEL_WIDTH>) profile_instance_.output_shape_.dims_[3]) << 32);
  stream_out.write (channel_out);
}

inline void Convolution_loadFilter (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    int * debug)
{
  Data temp_0;
  Data temp_1;
  volatile int filter_size = FlatSize (&profile_instance_.filter_shape_);
  *debug = 4;
  // CONV_LOAD_FILTER_LOOP
  CONV_LOAD_FILTER_LOOP: for (int i = 0; i < filter_size; i += 2)
  {
#pragma HLS pipeline
    channel_in = stream_in.read ();
    temp_0.u32 = channel_in.data;
    filter_instance_[i] = temp_0.f32;
    temp_1.u32 = channel_in.data >> 32;
    filter_instance_[i + 1] = temp_1.f32;
  }
}

inline void Convolution_loadBias (hls::stream<StreamChannel> &stream_in,
                                  hls::stream<StreamChannel> &stream_out,
                                  int * debug)
{
  Data temp_0;
  Data temp_1;
  volatile int bias_size = FlatSize (&profile_instance_.bias_shape_);
  *debug = 3;
  // CONV_LOAD_BIAS_LOOP
  CONV_LOAD_BIAS_LOOP: for (int i = 0; i < bias_size; i += 2)
  {
#pragma HLS pipeline
    channel_in = stream_in.read ();
    temp_0.u32 = channel_in.data;
    bias_instance_[i] = temp_0.f32;
    temp_1.u32 = channel_in.data >> 32;
    bias_instance_[i + 1] = temp_1.f32;
  }
}

inline void Convolution_fetchFilter (hls::stream<StreamChannel> &stream_in,
                                     hls::stream<StreamChannel> &stream_out,
                                     int * debug)
{
  Data temp_2;
  Data temp_3;
  volatile int filter_size = FlatSize (&profile_instance_.filter_shape_);
  *debug = 2;
  // CONV_LOAD_FILTER_LOOP
  CONV_FETCH_FILTER_LOOP: for (int i = 0; i < filter_size; i += 2)
  {
//#pragma HLS pipeline
    temp_2.f32 = filter_instance_[i];
    temp_3.f32 = filter_instance_[i + 1];

    channel_out.data = ((((ap_uint<64> ) temp_3.u32) << 32) & 0xFFFFFFFF00000000) | ((ap_uint<64> ) temp_2.u32);
    //channel_out.last = filter_size <= i + 2;
    stream_out.write (channel_out);
  }
}

inline void Convolution_fetchBias (hls::stream<StreamChannel> &stream_in,
                                   hls::stream<StreamChannel> &stream_out,
                                   int * debug)
{
  Data temp_2;
  Data temp_3;
  volatile int bias_size = FlatSize (&profile_instance_.bias_shape_);
  // CONV_LOAD_BIAS_LOOP
  *debug = 1;
  CONV_FETCH_BIAS_LOOP: for (int i = 0; i < bias_size; i += 2)
  {
//#pragma HLS pipeline
    temp_2.f32 = bias_instance_[i];
    temp_3.f32 = bias_instance_[i + 1];

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

//#pragma HLS DATAFLOW
//#pragma HLS INLINE region // bring loops in sub-functions to this DATAFLOW region

  int rc = 0;
  channel_out.keep = -1;
  channel_out.strb = -1;

  switch (mode)
  {
    case CONV_LOAD_PROFILE_PACKAGE:
      Convolution_loadProfile  (stream_in, stream_out, debug);
      Convolution_loadFilter   (stream_in, stream_out, debug);
      Convolution_loadBias     (stream_in, stream_out, debug);
      break;
//    case CONV_LOAD_PROFILE:
//      Convolution_loadProfile  (stream_in, stream_out, debug);
//      break;
    case CONV_FETCH_PROFILE:
      Convolution_fetchProfile (stream_in, stream_out, debug);
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
      Convolution_execution<float> (stream_in, stream_out, debug);
      break;
    default:
      rc = -1;
  }

  channel_out.last = 0;

  return rc;
}

