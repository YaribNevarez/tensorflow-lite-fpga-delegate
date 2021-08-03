/*
 * conv_delegate.h
 *
 *  Created on: July 31st, 2021
 *      Author: Yarib Nevarez
 */
#ifndef CONV_DELEGATE_H_
#define CONV_DELEGATE_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

#include "processing_unit.h"

class ConvFpgaDelegate: protected ProcessingUnit
{
public:

  ConvFpgaDelegate ();
  ~ConvFpgaDelegate ();

  virtual int initialize (void);

  virtual void Conv (const tflite::ConvParams& params,
        const tflite::RuntimeShape& input_shape, const float* input_data,
        const tflite::RuntimeShape& filter_shape, const float* filter_data,
        const tflite::RuntimeShape& bias_shape, const float* bias_data,
        const tflite::RuntimeShape& output_shape, float* output_data,
        const tflite::RuntimeShape& im2col_shape, float* im2col_data);

  virtual void ConvInternal (const tflite::ConvParams& params,
                const tflite::RuntimeShape& input_shape, const float* input_data,
                const tflite::RuntimeShape& filter_shape, const float* filter_data,
                const tflite::RuntimeShape& bias_shape, const float* bias_data,
                const tflite::RuntimeShape& output_shape, float* output_data,
                const tflite::RuntimeShape& im2col_shape, float* im2col_data);

};

#endif // CONV_DELEGATE_H_