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

ConvFpgaDelegate::ConvFpgaDelegate()
{

}

ConvFpgaDelegate::~ConvFpgaDelegate()
{

}

int ConvFpgaDelegate::initialize()
{
  ProcessingUnit::Profile profile = {
    .hwVtbl      = &HardwareVtbl_Conv_,
    .dmaVtbl     = &DMAHardwareVtbl_,
    .hwDeviceID    = XPAR_CONV_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXI_DMA_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_CONV_0_INTERRUPT_INTR,
    .dmaTxIntVecID = XPAR_FABRIC_AXI_DMA_0_MM2S_INTROUT_INTR,
    .dmaRxIntVecID = XPAR_FABRIC_AXI_DMA_0_S2MM_INTROUT_INTR,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x31000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x31FFFFFF,
      .blockIndex  = 0
    }
  };

  pu_.initialize(&profile);
  return 0;
}

//
//
//  // CONV_LOAD_PROFILE
//
//  ConvProfile conv_profile;
//
//  memset (&conv_profile, 0, sizeof(ConvProfile));
//
//  conv_profile.parameters_.stride_.height_ = 1;
//  conv_profile.parameters_.stride_.width_ = 1;
//
//  conv_profile.parameters_.dilation_.height_ = 1;
//  conv_profile.parameters_.dilation_.width_ = 1;
//
//  conv_profile.parameters_.padding_.height_ = 1;
//  conv_profile.parameters_.padding_.width_ = 1;
//
//  conv_profile.parameters_.activation_.max_ = std::numeric_limits<float>::infinity();
//  conv_profile.parameters_.activation_.min_ = 0;
//
//  conv_profile.input_shape_.dims_[0] = 1;
//  conv_profile.input_shape_.dims_[1] = 32;
//  conv_profile.input_shape_.dims_[2] = 32;
//  conv_profile.input_shape_.dims_[3] = 3;
//
//  conv_profile.filter_shape_.dims_[0] = 32;
//  conv_profile.filter_shape_.dims_[1] = 3;
//  conv_profile.filter_shape_.dims_[2] = 3;
//  conv_profile.filter_shape_.dims_[3] = 3;
//
//  conv_profile.filter_shape_.dims_[0] = 32;
//  conv_profile.filter_shape_.dims_[1] = 3;
//  conv_profile.filter_shape_.dims_[2] = 3;
//  conv_profile.filter_shape_.dims_[3] = 3;
//
//  conv_profile.bias_shape_.dims_[0] = 32;
//  conv_profile.bias_shape_.dims_[1] = 1;
//  conv_profile.bias_shape_.dims_[2] = 1;
//  conv_profile.bias_shape_.dims_[3] = 1;
//
//  conv_profile.output_shape_.dims_[0] = 1;
//  conv_profile.output_shape_.dims_[1] = 32;
//  conv_profile.output_shape_.dims_[2] = 32;
//  conv_profile.output_shape_.dims_[3] = 32;
//
//  //while (!Conv_hardware.IsReady (conv));
//
//  conv_flag_ = 0;
////
////  Conv_hardware.Set_mode (conv, CONV_LOAD_PROFILE);
////  Event_start (interpreter_event_);
////  Event_start (conv_sw_event_);
////  Event_start (conv_hw_event_);
////  Conv_hardware.Start (conv);
////
////  status = Dma_transaction (dmaHardware,
////                            &conv_profile, sizeof(conv_profile),
////                            nullptr, 0);
////
////  status = Conv_hardware.Get_debug(conv);
////
////  while (!Conv_hardware.IsDone (conv));
////  while (conv_flag_ < 1);
////
////  conv_flag_ = 0;
////
////  ConvProfile conv_profile_test = {0};
////
////  Conv_hardware.Set_mode (conv, CONV_FETCH_PROFILE);
////  Event_start (interpreter_event_);
////  Event_start (conv_sw_event_);
////  Event_start (conv_hw_event_);
////  Conv_hardware.Start (conv);
////
////  status = Dma_transaction (dmaHardware,
////                            nullptr, 0,
////                            &conv_profile_test, sizeof(conv_profile_test));
////
////  status = Conv_hardware.Get_debug(conv);
////
////  while (!Conv_hardware.IsDone (conv));
////  while (conv_flag_ < 1);
////
////
////
////  if (memcmp(&conv_profile, &conv_profile_test, sizeof(ConvProfile)))
////  {
////    printf ("Test fail !");
////  }
//
//
//
//  buffer_size = FlatSize(&conv_profile.bias_shape_) * sizeof (float);
//
//  float * tx_buffer = (float *) (baseAddress + buffer_size);
//  float * rx_buffer = (float *) baseAddress;
//
//  for (size_t i = 0; i < FlatSize(&conv_profile.filter_shape_); i ++)
//  {
//    tx_buffer[i] = float(i);
//  }
//
//  memset (rx_buffer, 0, buffer_size);
//  Xil_DCacheFlushRange ((UINTPTR) rx_buffer, buffer_size);
//
//
//  Conv_hardware.Set_mode (profile_->hwInstance, CONV_LOAD_BIAS);
//  Event_start (interpreter_event_);
//  Event_start (conv_sw_event_);
//  Event_start (conv_hw_event_);
//  Conv_hardware.Start (profile_->hwInstance);
//
//  status = Dma_transaction (profile_->dmaInstance, tx_buffer, buffer_size, nullptr, 0);
//
//  status = Conv_hardware.Get_debug(profile_->hwInstance);
//
//  while (!Conv_hardware.IsDone (profile_->hwInstance));
//  while (conv_flag_ < 1);
//
//  Conv_hardware.Set_mode (profile_->hwInstance, CONV_FETCH_BIAS);
//  Event_start (interpreter_event_);
//  Event_start (conv_sw_event_);
//  Event_start (conv_hw_event_);
//  Conv_hardware.Start (profile_->hwInstance);
//
//  status = Dma_transaction (profile_->dmaInstance, nullptr, 0, rx_buffer, buffer_size);
//
//  status = Conv_hardware.Get_debug(profile_->hwInstance);
//
//  while (!Conv_hardware.IsDone (profile_->hwInstance));
//  while (conv_flag_ < 1);
//
//  if (memcmp (tx_buffer, rx_buffer, buffer_size))
//  {
//    printf ("Test fail !");
//  }
//
//  Event_stop (conv_sw_event_);
//
//  Event_stop (interpreter_event_);
//
//  Event_print (interpreter_event_);
//
//  return status;
//}

void ConvFpgaDelegate::Conv(const tflite::ConvParams& params, const tflite::RuntimeShape& input_shape,
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
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
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
                float input_value = input_data[Offset(input_shape, batch, in_y,
                                                      in_x, in_channel)];
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

