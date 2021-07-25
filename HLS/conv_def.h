#ifndef CONV_DEF_H_
#define CONV_DEF_H_

#include <stdint.h>
#include <algorithm>

#define DMA_CHANNEL_WIDTH 64

typedef enum
{
  CONV_LOAD_PROFILE,
  CONV_FETCH_PROFILE,
  CONV_LOAD_FILTER,
  CONV_FETCH_FILTER,
  CONV_LOAD_BIAS,
  CONV_FETCH_BIAS,
  CONV_EXECUTION
} ConvExecutionMode;

typedef union
{
  unsigned int u32;
  float f32;
} Data;

#define MAX_TENSOR_SIZE 4
typedef struct
{
  int dims_[MAX_TENSOR_SIZE];
} TensorShape;

typedef struct
{
  Data * data_;
  TensorShape shape_;
} Tensor;

typedef struct
{
  int height_;
  int width_;
} ConvStride;

typedef struct
{
  int height_;
  int width_;
} ConvDilation;

typedef struct
{
  int height_;
  int width_;
} ConvPadding;

typedef struct
{
  float max_;
  float min_;
} ConvActivation;

typedef struct
{
  ConvStride      stride_;
  ConvDilation    dilation_;
  ConvPadding     padding_;
  ConvActivation  activation_;
} ConvParameters;


typedef struct
{
  ConvParameters  parameters_;
  TensorShape     input_shape_;
  TensorShape     filter_shape_;
  TensorShape     bias_shape_;
  TensorShape     output_shape_;
} ConvProfile;


#endif // CONV_DEF_H_

