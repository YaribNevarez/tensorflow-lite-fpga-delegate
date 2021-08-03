/*
 * conv_delegate.cpp
 *
 *  Created on: July 31st, 2021
 *      Author: Yarib Nevarez
 */

#include "conv_delegate.h"
#include "tensorflow/lite/kernels/internal/common.h"

#include "conv_vtbl.h"
#include "dma_vtbl.h"
#include "conv_hls.h"
#include "memory_manager.h"

ConvFpgaDelegate::ConvFpgaDelegate()
{

}

ConvFpgaDelegate::~ConvFpgaDelegate()
{

}

int ConvFpgaDelegate::initialize()
{
  static ProcessingUnit::Profile profile = {
    .hwVtbl        = &HardwareVtbl_Conv_,
    .dmaVtbl       = &DMAHardwareVtbl_,
    .hwDeviceID    = XPAR_CONV_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXI_DMA_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_CONV_0_INTERRUPT_INTR,
    .dmaTxIntVecID = XPAR_FABRIC_AXI_DMA_0_MM2S_INTROUT_INTR,
    .dmaRxIntVecID = XPAR_FABRIC_AXI_DMA_0_S2MM_INTROUT_INTR,
    .channelSize   = 8,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x31000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x31FFFFFF,
      .blockIndex  = 0
    }
  };

  return ProcessingUnit::initialize (&profile);
}

void ConvFpgaDelegate::Conv (const tflite::ConvParams& params,
                             const tflite::RuntimeShape& input_shape,
                             const float* input_data,
                             const tflite::RuntimeShape& filter_shape,
                             const float* filter_data,
                             const tflite::RuntimeShape& bias_shape,
                             const float* bias_data,
                             const tflite::RuntimeShape& output_shape,
                             float* output_data,
                             const tflite::RuntimeShape& im2col_shape,
                             float* im2col_data)
{
  static Transaction transaction = { 0 };
  size_t txBufferSize = 0;
  size_t rxBufferSize = 0;
  void * txBufferPtr = nullptr;
  void * rxBufferPtr = nullptr;
  ConvProfile * conv_profile = nullptr;
  float * filter = nullptr;
  float * bias = nullptr;

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount (), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount (), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount (), 4);

  if (bias_data)
  {
    const int output_depth = MatchingDim (filter_shape, 0, output_shape, 3);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize (), output_depth);
  }

  txBufferSize = sizeof(ConvProfile) + (filter_shape.FlatSize () + bias_shape.FlatSize ()) * sizeof(float);
  txBufferPtr = MemoryBlock_alloc (&profile_->ddrMem, txBufferSize);
  memset (txBufferPtr, 0, txBufferSize);

  transaction.mode = CONV_LOAD_PROFILE_PACKAGE;
  transaction.flags = BLOCKING_IN_OUT | RX_CACHE_FETCH | TX_CACHE_FUSH;
  transaction.txBufferPtr = txBufferPtr;
  transaction.txBufferSize = txBufferSize;
  transaction.rxBufferPtr = nullptr;
  transaction.rxBufferSize = 0;

  conv_profile = (ConvProfile *) txBufferPtr;
  filter = (float *) &conv_profile[1];
  bias = &filter[filter_shape.FlatSize ()];

  conv_profile->parameters_.stride_.height_ = params.stride_height;
  conv_profile->parameters_.stride_.width_ = params.stride_width;

  conv_profile->parameters_.dilation_.height_ = params.dilation_height_factor;
  conv_profile->parameters_.dilation_.width_ = params.dilation_width_factor;

  conv_profile->parameters_.padding_.height_ = params.padding_values.height;
  conv_profile->parameters_.padding_.width_ = params.padding_values.width;

  conv_profile->parameters_.activation_.max_ = params.float_activation_max;
  conv_profile->parameters_.activation_.min_ = params.float_activation_min;

  conv_profile->input_shape_.dims_[0] = input_shape.Dims (0);
  conv_profile->input_shape_.dims_[1] = input_shape.Dims (1);
  conv_profile->input_shape_.dims_[2] = input_shape.Dims (2);
  conv_profile->input_shape_.dims_[3] = input_shape.Dims (3);

  conv_profile->filter_shape_.dims_[0] = filter_shape.Dims (0);
  conv_profile->filter_shape_.dims_[1] = filter_shape.Dims (1);
  conv_profile->filter_shape_.dims_[2] = filter_shape.Dims (2);
  conv_profile->filter_shape_.dims_[3] = filter_shape.Dims (3);

  conv_profile->bias_shape_.dims_[0] = bias_shape.Dims (0);
  conv_profile->bias_shape_.dims_[1] = 1;
  conv_profile->bias_shape_.dims_[2] = 1;
  conv_profile->bias_shape_.dims_[3] = 1;

  conv_profile->output_shape_.dims_[0] = output_shape.Dims (0);
  conv_profile->output_shape_.dims_[1] = output_shape.Dims (1);
  conv_profile->output_shape_.dims_[2] = output_shape.Dims (2);
  conv_profile->output_shape_.dims_[3] = output_shape.Dims (3);

  memcpy (filter,
          filter_data,
          filter_shape.FlatSize () * sizeof(float));

  memcpy (bias,
          bias_data,
          bias_shape.FlatSize () * sizeof(float));

  //execute (&transaction);


  rxBufferSize = sizeof(ConvProfile);
  rxBufferPtr = MemoryBlock_alloc (&profile_->ddrMem, rxBufferSize);
  memset (rxBufferPtr, 0, rxBufferSize);

  transaction.mode = CONV_FETCH_PROFILE;
  transaction.flags = BLOCKING_IN_OUT | RX_CACHE_FETCH | TX_CACHE_FUSH;
  transaction.txBufferPtr = nullptr;
  transaction.txBufferSize = 0;
  transaction.rxBufferPtr = rxBufferPtr;
  transaction.rxBufferSize = rxBufferSize;
  //execute (&transaction);

  ConvInternal (params, input_shape, input_data, filter_shape, filter_data,
                bias_shape, bias_data, output_shape, output_data, im2col_shape,
                im2col_data);
}

static float StreamPeripheral_inputBuffer[1024*1024] = {0};
static int StreamPeripheral_yOffset = 0;
static int StreamPeripheral_lookupTable[32] = {0};
static int StreamPeripheral_lookupIndex = 0;
static int StreamPeripheral_lookupLength = 0;

static int AXIStream_index = 0;


//inline int Offset(const RuntimeShape& shape, int i0, int i1, int i2, int i3) {
//  TFLITE_DCHECK_EQ(shape.DimensionsCount(), 4);
//  const int* dims_data = reinterpret_cast<const int*>(shape.DimsDataUpTo5D());
//  TFLITE_DCHECK(i0 >= 0 && i0 < dims_data[0]);
//  TFLITE_DCHECK(i1 >= 0 && i1 < dims_data[1]);
//  TFLITE_DCHECK(i2 >= 0 && i2 < dims_data[2]);
//  TFLITE_DCHECK(i3 >= 0 && i3 < dims_data[3]);
//  return ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3;
//}

float StreamPeripheral_inputData (const tflite::RuntimeShape& input_shape,
                                  int batch,
                                  int in_y,
                                  int in_x,
                                  int in_channel)
{
  int lookupIndex;
  int i;

  lookupIndex = in_y - StreamPeripheral_yOffset + StreamPeripheral_lookupIndex;

  if (lookupIndex > StreamPeripheral_lookupLength)
    lookupIndex -= StreamPeripheral_lookupLength;

  i = StreamPeripheral_lookupTable[lookupIndex] + input_shape.Dims(3)*in_x + in_channel;

  return StreamPeripheral_inputBuffer[i];
}

void StreamPeripheral_outputData (TensorShape output_shape,
                                  int batch,
                                  int out_y,
                                  int out_x,
                                  int out_channel)
{

}

void StreamPeripheral_loadSlice (const float * input_data,
                                 int input_depth,
                                 int input_width,
                                 int filter_height)
{
  int i = 0;
  AXIStream_index = 0;

  StreamPeripheral_lookupLength = filter_height;

  StreamPeripheral_lookupIndex = StreamPeripheral_lookupLength - 1;

  for (int row = 0; row < StreamPeripheral_lookupIndex; row++)
  {
    StreamPeripheral_lookupTable[row] = i;
    for (int col = 0; col < input_width; col++)
    {
      for (int chan = 0; chan < input_depth; chan++)
      {
        StreamPeripheral_inputBuffer[i] = input_data[AXIStream_index ++];
        i++;
      }
    }
  }

  StreamPeripheral_lookupTable[StreamPeripheral_lookupIndex] = i;
}

//static float StreamPeripheral_inputBuffer[1024] = {0};
//static int StreamPeripheral_bufferVectorLength = 0;
//static int StreamPeripheral_lookupTable[32] = {0};
//static int StreamPeripheral_stateIndex = 0;

void StreamPeripheral_pushSlice (const float * input_data,
                                  int input_depth,
                                  int input_width,
                                  int filter_height,
                                  int in_y_origin)
{
  int i = StreamPeripheral_lookupTable[StreamPeripheral_lookupIndex];

  StreamPeripheral_yOffset = in_y_origin;

  if (0 <= StreamPeripheral_yOffset)
  {
    for (int col = 0; col < input_width; col++)
    {
      for (int chan = 0; chan < input_depth; chan++)
      {
        StreamPeripheral_inputBuffer[i] = input_data[AXIStream_index++];
        i++;
      }
    }

    if (StreamPeripheral_lookupIndex < filter_height - 1)
    {
      StreamPeripheral_lookupIndex++;
    }
    else
    {
      StreamPeripheral_lookupIndex = 0;
    }
  }
}

void ConvFpgaDelegate::ConvInternal(const tflite::ConvParams& params, const tflite::RuntimeShape& input_shape,
                 const float* input_data, const tflite::RuntimeShape& filter_shape,
                 const float* filter_data, const tflite::RuntimeShape& bias_shape,
                 const float* bias_data, const tflite::RuntimeShape& output_shape,
                 float* output_data, const tflite::RuntimeShape& im2col_shape,
                 float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  StreamPeripheral_loadSlice (input_data,
                              input_depth,
                              input_width,
                              filter_height);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;

      StreamPeripheral_pushSlice (input_data,
                                  input_depth,
                                  input_width,
                                  filter_height,
                                  in_y_origin);

      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                float input_value = StreamPeripheral_inputData(input_shape, batch, in_y,
                                                      in_x, in_channel);
                float filter_value = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                total += (input_value * filter_value);
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              tflite::ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}
