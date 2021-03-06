/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 1024 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

extern unsigned char cifar_cnn_tflite[];

// The name of this function is important for Arduino compatibility.
void setup ()
{
  tflite::InitializeTarget ();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel (cifar_cnn_tflite);
  if (model->version () != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version (), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver micro_op_resolver;

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter (model, micro_op_resolver,
                                                      tensor_arena,
                                                      kTensorArenaSize,
                                                      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors ();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input (0);

  TF_LITE_REPORT_ERROR(error_reporter, "input->dims->size = %d",
                       input->dims->size);
  for (int i = 0; i < input->dims->size; i++)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "input->dims->data[%d] = %d", i,
                         input->dims->data[i]);
  }
  TF_LITE_REPORT_ERROR(error_reporter, "input->type = 0x%d", input->type);

//////////////////////////////////////////////////////////////////////////////////
  TfLiteTensor* output = interpreter->output (0);

  TF_LITE_REPORT_ERROR(error_reporter, "output->dims->size = %d",
                       output->dims->size);
  for (int i = 0; i < output->dims->size; i++)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "output->dims->data[%d] = %d", i,
                         output->dims->data[i]);
  }
  TF_LITE_REPORT_ERROR(error_reporter, "input->type = 0x%d", output->type);
}

// The name of this function is important for Arduino compatibility.
void loop ()
{
  TfLiteStatus status;
  static int image_index = 0;

  if (10 <= image_index)
  {
    image_index = 0;
  }

  status = GetImage (error_reporter,
                     image_index,
                     input->dims->data[1],
                     input->dims->data[2],
                     input->dims->data[3],
                     input->data.f);

  if (status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }


  status = interpreter->Invoke ();

  if (status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }


  interpreter->get_eventLog ();

  TfLiteTensor* output = interpreter->output (0);

  RespondToDetection (error_reporter, output, image_index);

  image_index++;
}
