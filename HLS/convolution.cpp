#include "convolution.h"
#include "../libs/utilities/inc/custom_float.h"
#include <stdint.h>
#include <algorithm>

#define HYBRID_LOGARITHMIC      true
#define LOG_FORMAT_BIT_WIDTH    8
#define LOG_MANTISSA_BIT_WIDTH  3

#if HYBRID_LOGARITHMIC
typedef uint8_t         CustomFormat;
typedef int64_t         MagnitudeFormat;
typedef int8_t          ExponentFormat;
typedef bool            SignFormat;
#else
typedef float           CustomFormat;
#endif


#define DMA_CHANNEL_BYTE_WIDTH    (DMA_CHANNEL_WIDTH/8)
#define DMA_CHANNEL_FLOAT_WIDTH   (DMA_CHANNEL_BYTE_WIDTH/sizeof(float))


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

#if HYBRID_LOGARITHMIC
inline MagnitudeFormat Float_denormalize (float value)
{
  Data data;
  SignFormat sign = 0;
  ExponentFormat exponent = 0;
  MagnitudeFormat mantissa = 0;
  MagnitudeFormat magnitude = 0;

  data.f32 = value;

  if (data.u32)
  {
    sign = DATA32_GET_SIGN(data.u32);
    exponent = DATA32_GET_EXPONENT(data.u32);
    mantissa = DATA32_GET_MANTISSA(data.u32);
    magnitude =
        (0 < exponent) ? (mantissa << exponent) : (mantissa >> -exponent);

    if (32 < exponent)
      magnitude = (((MagnitudeFormat)(1)) << 63) - 1;

    if (sign)
      magnitude = ~magnitude + 1;
  }

  return magnitude;
}

inline MagnitudeFormat Custom_denormalize (CustomFormat value)
{
  SignFormat sign = 0;
  ExponentFormat exponent = 0;
  MagnitudeFormat mantissa = 0;
  MagnitudeFormat magnitude = 0;

  if (value)
  {
    sign = value & (1 << (LOG_FORMAT_BIT_WIDTH - 1));
    exponent = -((value & ((1 << (LOG_FORMAT_BIT_WIDTH - 1)) - 1))
        >> LOG_MANTISSA_BIT_WIDTH);
    mantissa = 0x00800000
        | ((value & ((1 << LOG_MANTISSA_BIT_WIDTH) - 1))
            << (23 - LOG_MANTISSA_BIT_WIDTH));

    magnitude =
        (0 < exponent) ? (mantissa << exponent) : (mantissa >> -exponent);

    if (sign)
      magnitude = ~magnitude + 1;
  }

  return magnitude;
}

inline MagnitudeFormat ActivationFunctionWithMinMaxMagnitude (MagnitudeFormat x,
                                       MagnitudeFormat output_activation_min,
                                       MagnitudeFormat output_activation_max)
{

  return (x < output_activation_min) ? output_activation_min :
         (output_activation_max < x) ? output_activation_max : x;
}

inline void DotProduct_logarithmic (MagnitudeFormat & Total_magnitude,
                                    float & input_value,
                                    CustomFormat & filter_value)
{
  Data      input_data;

  SignFormat        f_s;
  ExponentFormat    f_e;
  MagnitudeFormat   f_m;

  SignFormat        i_s;
  ExponentFormat    i_e;
  MagnitudeFormat   i_m;

  SignFormat        p_s;
  ExponentFormat    p_e;
  MagnitudeFormat   p_m;

  MagnitudeFormat   p_magnitude;

  input_data.f32 = input_value;

  if (input_data.u32 == 0 || filter_value == 0)
    return;

  i_s = DATA32_GET_SIGN(input_data.u32);
  i_e = DATA32_GET_EXPONENT(input_data.u32);
  i_m = DATA32_GET_MANTISSA(input_data.u32);

  f_s = filter_value & (1 << (LOG_FORMAT_BIT_WIDTH - 1));
  f_e = -((filter_value & ((1 << (LOG_FORMAT_BIT_WIDTH - 1)) - 1))
      >> LOG_MANTISSA_BIT_WIDTH);
  f_m = 0x00800000
      | ((filter_value & ((1 << LOG_MANTISSA_BIT_WIDTH) - 1)) << (23 - LOG_MANTISSA_BIT_WIDTH));

  p_s = i_s != f_s;
  p_e = i_e + f_e;
  p_m = (i_m * f_m) >> 23;

  if (p_m & 0x01000000)
  {
    p_e++;
    p_m >>= 1;
  }

  p_magnitude = (0 < p_e) ? (p_m << p_e) : (p_m >> -p_e);

  if (p_s)
    p_magnitude = ~p_magnitude + 1;

  Total_magnitude += p_magnitude;
}
#endif

static int Convolution_execution (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    ConvProfile &  Conv_profile,
                                    CustomFormat * Conv_filter,
                                    CustomFormat * Conv_bias,
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
#if HYBRID_LOGARITHMIC
  SignFormat        Total_sign = 0;
  ExponentFormat    Total_exponent = 0;
  MagnitudeFormat   Total_magnitude = 0;
  ///////////////////////////////////////////////////////////////////////////////
  MagnitudeFormat   Activation_max_magnitude = 0;
  MagnitudeFormat   Activation_min_magnitude = 0;
#endif
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

  TensorShape * output_shape = &profile->output_shape_;
  volatile int output_height = output_shape->dims_[1];
  volatile int output_width = output_shape->dims_[2];
  volatile int output_depth = output_shape->dims_[3];

  TensorShape * bias_shape = &profile->bias_shape_;

  bool bias_data_enable = (bias_shape->dims_[0] == output_depth);

  int stride_height = profile->parameters_.stride_.height_;
  int stride_width = profile->parameters_.stride_.width_;

  int pad_height = profile->parameters_.padding_.height_;
  int pad_width = profile->parameters_.padding_.width_;

  int dilation_height_factor = profile->parameters_.dilation_.height_;
  int dilation_width_factor = profile->parameters_.dilation_.height_;

  float output_activation_max = profile->parameters_.activation_.max_;
  float output_activation_min = profile->parameters_.activation_.min_;

  float activation_output = 0;

  ///////////////////////////////////////////////////////////////////////////////
#if HYBRID_LOGARITHMIC
  Activation_max_magnitude = Float_denormalize (output_activation_max);
  Activation_min_magnitude = Float_denormalize (output_activation_min);
#endif
  ///////////////////////////////////////////////////////////////////////////////
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

    INITIALIZE_INPUT_TENSOR_LINE_ROWS: for (int row = 0; row < StreamPeripheral_lookupIndex; row++)
    {
#pragma HLS pipeline
      StreamPeripheral_lookupTable[row] = AXIStream_index;
      StreamPeripheral_lookupTableRows[row] = StreamPeripheral_rowCount++;

      INITIALIZE_INPUT_TENSOR_LINE_ROW: for (int j = 0; j < StreamPeripheral_rowSize; j += 2)
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

          READ_INPUT_TENSOR_LINE_ROW: for (int j = 0; j < StreamPeripheral_rowSize; j += 2)
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
#if HYBRID_LOGARITHMIC
          Total_magnitude = 0;
#endif
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

                CustomFormat filter_value = Conv_filter[Offset (filter_shape,
                                                         out_channel, filter_y,
                                                         filter_x, in_channel)];
#if HYBRID_LOGARITHMIC
                  DotProduct_logarithmic (Total_magnitude, input_value, filter_value);
#else
                  total += (input_value * filter_value);
#endif
              }
            }
          }

#if HYBRID_LOGARITHMIC
          if (bias_data_enable)
          {
              Total_magnitude += Custom_denormalize (Conv_bias[out_channel]);
          }

          Total_magnitude = ActivationFunctionWithMinMaxMagnitude (Total_magnitude,
                                                            Activation_min_magnitude,
                                                            Activation_max_magnitude);

          if (Total_magnitude != 0)
          {
            Total_sign = Total_magnitude < 0;
            if (Total_sign)
            {
              Total_magnitude = ~Total_magnitude + 1;
            }

            TOTAL_NORMALIZATION_LOOP: for (Total_exponent = 0; !(0x80000000000000 & Total_magnitude); Total_exponent++)
            { // Normalize
#pragma HLS pipeline
              Total_magnitude <<= 1;
            }

            Total_exponent -= 32;
            Total_magnitude >>= 32;

            *(uint32_t*) (&activation_output) = BUILD_FLOAT(Total_sign, -Total_exponent, Total_magnitude);
          }
          else
          {
            *(uint32_t*) (&activation_output) = 0;
          }
#else
          float bias_value = 0.0f;
          if (bias_data_enable)
          {
            bias_value = Conv_bias[out_channel];
          }
          total += bias_value;

          activation_output = ActivationFunctionWithMinMax (total,
                                                            output_activation_min,
                                                            output_activation_max);
#endif

          /*out()*/
          {
            AXIStream_outputBuffer[AXIStreamOut_index % 2] = activation_output;
            if (AXIStreamOut_index % 2 == 2 - 1)
            {
              Data temp_0;
              Data temp_1;
              temp_0.f32 = AXIStream_outputBuffer[0];
              temp_1.f32 = AXIStream_outputBuffer[1];
              channel_out.data = ((ap_uint<DMA_CHANNEL_WIDTH> ) temp_0.u32)
                  | (((ap_uint<DMA_CHANNEL_WIDTH> ) temp_1.u32) << 32);
              channel_out.last = (AXIStreamOut_index + 1)
                  == AXIStreamOut_indexLast;
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

inline void Convolution_loadTensor (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    TensorShape * tensorShape,
                                    CustomFormat * tensor,
                                    int * debug)
{
#if HYBRID_LOGARITHMIC
  ExponentFormat     exponent[DMA_CHANNEL_FLOAT_WIDTH];
  MagnitudeFormat    mantissa[DMA_CHANNEL_FLOAT_WIDTH];
#endif

  Data temp[DMA_CHANNEL_FLOAT_WIDTH];

  volatile int tensor_size = FlatSize (tensorShape);
  *debug = 4;
  // CONV_LOAD_FILTER_LOOP
  CONV_LOAD_TENSOR_LOOP: for (int i = 0; i < tensor_size; i += DMA_CHANNEL_FLOAT_WIDTH)
  {
#pragma HLS pipeline
    channel_in = stream_in.read ();

    for (int j = 0; j < DMA_CHANNEL_FLOAT_WIDTH; j++)
    {
#pragma HLS unroll
      temp[j].u32 = channel_in.data >> (8 * sizeof(float) * j);

#if HYBRID_LOGARITHMIC
      exponent[j] = DATA32_GET_EXPONENT(temp[j].u32);
      mantissa[j] = DATA32_GET_MANTISSA(temp[j].u32);

      if ((LOG_MANTISSA_BIT_WIDTH == 0) && (0x400000 < (0x7FFFFF & mantissa[j])))
      {
        exponent[j]++;
      }

      if (exponent[j] < - ((1 << (LOG_FORMAT_BIT_WIDTH - LOG_MANTISSA_BIT_WIDTH - 1)) - 1))
      {
        tensor[i + j] = 0;
      }
      else
      {
        exponent[j] = ~exponent[j] + 1;

        tensor[i + j] = ((1 << (LOG_FORMAT_BIT_WIDTH - 1)) & (temp[j].u32 >> (32 - LOG_FORMAT_BIT_WIDTH)))
                        | (exponent[j] << LOG_MANTISSA_BIT_WIDTH)
                        | ((0x7FFFFF & mantissa[j]) >> (23 - LOG_MANTISSA_BIT_WIDTH));
      }

#else
      tensor[i + j] = temp[j].f32;
#endif
    }
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
  #define CONV_FILTER_BUFFER_SIZE (256*1024)
  #define CONV_BIAS_BUFFER_SIZE   (256)

  static ConvProfile  Conv_profile;
  static CustomFormat Conv_filter[CONV_FILTER_BUFFER_SIZE];
  static CustomFormat Conv_bias[CONV_BIAS_BUFFER_SIZE];
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
      Convolution_loadTensor   (stream_in, stream_out, &Conv_profile.filter_shape_, Conv_filter, debug);
      Convolution_loadTensor   (stream_in, stream_out, &Conv_profile.bias_shape_, Conv_bias, debug);
      break;
    case CONV_EXECUTION:
      Convolution_execution (stream_in, stream_out, Conv_profile, Conv_filter, Conv_bias, debug);
      break;
    default:
      rc = -1;
  }

  channel_out.last = 0;

  return rc;
}

