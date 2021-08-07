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
#include "miscellaneous.h"

void Buffer_print (void * data, size_t size, char* name)
{
  printf ("unsigned int %s [] = {", name);
  for (int i = 0, c = 0; i < size/sizeof(unsigned int); i ++)
  {
    printf ("0x%X%s", ((unsigned int*) data)[i], (i + 1 < size/sizeof(unsigned int))?", ":"");
    if (++c == 8)
    {
      c = 0;
      printf ("\n");
    }
  }
  printf ("};\nunsigned int %s_len = %d;\n", name,
          size / sizeof(unsigned int));
}

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

ConvFpgaDelegate::NodeProfile ConvFpgaDelegate::GenNodeProfile (const tflite::ConvParams& params,
                             const tflite::RuntimeShape& input_shape,
                             const float* input_data,
                             const tflite::RuntimeShape& filter_shape,
                             const float* filter_data,
                             const tflite::RuntimeShape& bias_shape,
                             const float* bias_data,
                             const tflite::RuntimeShape& output_shape,
                             float* output_data,
                             const tflite::RuntimeShape& im2col_shape,
                             float* im2col_data,
                             Event * parent)
{
  NodeProfile nodeSettings = { 0 };
  size_t txBufferSize = 0;
  void * txBufferPtr = nullptr;
  ConvProfile * conv_profile = nullptr;
  float * filter = nullptr;
  float * bias = nullptr;

  if (73728 < filter_shape.FlatSize ())
  {
    return nodeSettings;
  }

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount (), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount (), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount (), 4);

  if (bias_data)
  {
    const int output_depth = MatchingDim (filter_shape, 0, output_shape, 3);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize (), output_depth);
  }

  txBufferSize = sizeof(ConvProfile) + (filter_shape.FlatSize () + bias_shape.FlatSize ()) * sizeof(float);

  txBufferPtr = MemoryBlock_alloc (&profile_->ddrMem, 1024 * 1024);
  memset (txBufferPtr, 0, 1024 * 1024);
  Xil_DCacheFlushRange ((UINTPTR) txBufferPtr, 1024 * 1024);

  nodeSettings.setup.mode = CONV_LOAD_PROFILE_PACKAGE;
  nodeSettings.setup.flags = BLOCKING_IN_OUT | RX_CACHE_FETCH | TX_CACHE_FUSH;
  nodeSettings.setup.txBufferPtr = txBufferPtr;
  nodeSettings.setup.txBufferSize = txBufferSize;
  nodeSettings.setup.rxBufferPtr = nullptr;
  nodeSettings.setup.rxBufferSize = 0;

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

  nodeSettings.compute.mode = CONV_EXECUTION;
  nodeSettings.compute.flags = BLOCKING_IN_OUT | RX_CACHE_FETCH | TX_CACHE_FUSH;
  nodeSettings.compute.txBufferPtr = (void *) input_data;
  nodeSettings.compute.txBufferSize = input_shape.FlatSize() * sizeof(float);
  nodeSettings.compute.rxBufferPtr = (void *) output_data;
  nodeSettings.compute.rxBufferSize = output_shape.FlatSize() * sizeof(float);

  nodeSettings.event = Event_new (parent, EVENT_HARDWARE, (void *) "CONV_HW");

  return nodeSettings;
}

int ConvFpgaDelegate::execute(NodeProfile * profile)
{
  int status = XST_FAILURE;
  ASSERT(profile != nullptr);

  if (profile != nullptr)
  {
    status = ProcessingUnit::execute(&profile->setup);
    ASSERT(status == XST_SUCCESS);

    Event_start (profile->event);

    status = ProcessingUnit::execute(&profile->compute);
    ASSERT(status == XST_SUCCESS);

    Event_stop (profile->event);
  }

  return status;
}

void ConvFpgaDelegate::onDone_ip (void)
{

}

void ConvFpgaDelegate::onDone_dmaRx (void)
{

}

void ConvFpgaDelegate::onDone_dmaTx (void)
{

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
  static void * txBufferPtr = nullptr;
  void * rxBufferPtr = nullptr;
  ConvProfile * conv_profile = nullptr;
  float * filter = nullptr;
  float * bias = nullptr;

  if (73728 < filter_shape.FlatSize ())
  {
    ConvInternal (params, input_shape, input_data, filter_shape, filter_data,
                  bias_shape, bias_data, output_shape, output_data,
                  im2col_shape, im2col_data);
    printf("Bypass, filter bigger than 73728\n");
    return;
  }

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount (), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount (), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount (), 4);

  if (bias_data)
  {
    const int output_depth = MatchingDim (filter_shape, 0, output_shape, 3);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize (), output_depth);
  }

  txBufferSize = sizeof(ConvProfile) + (filter_shape.FlatSize () + bias_shape.FlatSize ()) * sizeof(float);

  if (txBufferPtr == nullptr)
  {
    txBufferPtr = MemoryBlock_alloc (&profile_->ddrMem, 1024 * 1024);
    memset (txBufferPtr, 0, 1024 * 1024);
    Xil_DCacheFlushRange ((UINTPTR) txBufferPtr, 1024 * 1024);
  }

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

  ProcessingUnit::execute (&transaction);

  size_t input_data_buffer_size = input_shape.FlatSize() * sizeof(float);
  static float * input_data_buffer = nullptr;

  if (input_data_buffer == nullptr)
  {
    input_data_buffer = (float*) MemoryBlock_alloc (&profile_->ddrMem,
                                                    1024 * 1024);
    memset (input_data_buffer, 0, 1024 * 1024);
    Xil_DCacheFlushRange ((UINTPTR) input_data_buffer, 1024 * 1024);
  }

  size_t output_data_buffer_size = output_shape.FlatSize() * sizeof(float);
  static float * output_data_buffer = nullptr;

  if (output_data_buffer == nullptr)
  {
    output_data_buffer = (float*) MemoryBlock_alloc (&profile_->ddrMem,
                                                     1024 * 1024);

    memset (output_data_buffer, 0, 1024 * 1024);
    Xil_DCacheFlushRange ((UINTPTR) output_data_buffer, 1024 * 1024);
  }

  memcpy (input_data_buffer, input_data, input_data_buffer_size);

  transaction.mode = CONV_EXECUTION;
  transaction.flags = BLOCKING_IN_OUT | RX_CACHE_FETCH | TX_CACHE_FUSH;
  transaction.txBufferPtr = (void *) input_data;
  transaction.txBufferSize = input_data_buffer_size;
  transaction.rxBufferPtr = (void *) output_data;
  transaction.rxBufferSize = output_data_buffer_size;
  ProcessingUnit::execute (&transaction);

  ConvInternal (params, input_shape, input_data, filter_shape, filter_data,
                bias_shape, bias_data, output_shape, output_data_buffer, im2col_shape,
                im2col_data);

  if(0 == memcmp(output_data_buffer, output_data, output_data_buffer_size))
  {
    printf("Processing Unit [Pass]!\n");
  }
  else
  {
    printf("Processing Unit [Fail]!\n");
  }
}

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

float StreamPeripheral_inputData (const tflite::RuntimeShape& input_shape,
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

  i = StreamPeripheral_lookupTable[lookupIndex] + input_shape.Dims(3)*in_x + in_channel;

  return StreamPeripheral_inputBuffer[i];
}

void StreamPeripheral_outputData (float * output_data, float output)
{
  AXIStream_outputBuffer[AXIStreamOut_index % 2] = output;
  if (AXIStreamOut_index % 2 == 2 - 1)
  {
    output_data[AXIStreamOut_index - 1] = AXIStream_outputBuffer[0];
    output_data[AXIStreamOut_index - 0] = AXIStream_outputBuffer[1];
  }
  AXIStreamOut_index ++;
}

void StreamPeripheral_initialize (const float * input_data,
                                 int input_depth,
                                 int input_width,
                                 int filter_height)
{
  AXIStream_index = 0;

  AXIStreamOut_index = 0;

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
          AXIStream_inputBuffer[0] = input_data[AXIStream_index + 0];
          AXIStream_inputBuffer[1] = input_data[AXIStream_index + 1];
        }
        StreamPeripheral_inputBuffer[AXIStream_index] = AXIStream_inputBuffer[AXIStream_index%2];
        AXIStream_index++;
      }
    }
  }

  StreamPeripheral_lookupTable[StreamPeripheral_lookupIndex] = AXIStream_index;

  ASSERT (StreamPeripheral_lookupLength == filter_height);
  ASSERT (StreamPeripheral_lookupIndex == StreamPeripheral_lookupLength - 1);
  ASSERT (StreamPeripheral_lookupIndex == StreamPeripheral_rowCount);
  ASSERT (AXIStream_index == StreamPeripheral_rowCount * input_depth * input_width);
}

void StreamPeripheral_pushSlice (const float * input_data,
                                  int input_depth,
                                  int input_width,
                                  int input_height,
                                  int filter_height,
                                  int in_y_origin)
{
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
          AXIStream_inputBuffer[0] = input_data[AXIStream_index + 0];
          AXIStream_inputBuffer[1] = input_data[AXIStream_index + 1];
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

  StreamPeripheral_initialize (input_data,
                              input_depth,
                              input_width,
                              filter_height);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;

      StreamPeripheral_pushSlice (input_data,
                                  input_depth,
                                  input_width,
                                  input_height,
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

                ASSERT (input_value == input_data[Offset (input_shape, batch, in_y, in_x,
                                          in_channel)]);

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
          StreamPeripheral_outputData(output_data,
              tflite::ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max));
        }
      }
    }
  }
}
