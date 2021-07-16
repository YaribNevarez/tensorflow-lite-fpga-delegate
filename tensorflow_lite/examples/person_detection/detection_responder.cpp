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

#include "detection_responder.h"
#include "model_settings.h"

#include "stdlib.h"
#include "string.h"
#include "stdio.h"

// This dummy implementation writes person and no person scores to the error
// console. Real applications will want to take some custom action instead, and
// should implement their own versions of this function.
void RespondToDetection (tflite::ErrorReporter* error_reporter,
                         TfLiteTensor* output,
                         int expected_index)
{
  char message[80] = {0};
  float temp;
  int index = 0;

  TF_LITE_REPORT_ERROR(error_reporter, "\nTensorflow lite CNN CIFAR classificator");

  TF_LITE_REPORT_ERROR(error_reporter, "Output tensor:");

  temp = 0;
  for (int i = 0; i < output->dims->data[1]; i++)
  {
    if (temp < output->data.f[i])
    {
      temp = output->data.f[i];
      index = i;
    }

    sprintf(message, "%f [%s]", output->data.f[i], CifarClassLabels[i]);
    TF_LITE_REPORT_ERROR(error_reporter, message);
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Classification: %s (%s)\n",
                       CifarClassLabels[index],
                       (expected_index == index) ? "PASS" : "FAIL");
}
