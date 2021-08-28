#include <iostream>
#include <cstdlib>

#include "convolution.h"
#include "data_test.h"
#include "cmath"

using namespace std;

namespace conv_test
{

typedef struct
{
  const char *    name;

  unsigned int *  transaction_setup;
  unsigned int    transaction_setup_len;

  unsigned int *  input_tensor;
  unsigned int    input_tensor_len;

  unsigned int *  output_tensor;
  unsigned int    output_tensor_len;
} TestCase;

static void Stream_setData(hls::stream<StreamChannel> & stream, unsigned int * data, unsigned int len)
{
  StreamChannel channel;
  for (unsigned int i = 0; i < len; i += 2)
  {
    channel.data = (0xFFFFFFFF00000000 & (((ap_uint<DMA_CHANNEL_WIDTH> ) data[i + 1]) << 32))
              | (0x00000000FFFFFFFF & data[i]);
    stream.write (channel);
  }
}

static void setup (unsigned int * transaction_setup,
                   unsigned int transaction_setup_len)
{
  int debug;
  // Send setup transaction
  printf("_________ Setup _________\n");

  hls::stream<StreamChannel> stream_transaction ("transaction");
  hls::stream<StreamChannel> stream_out ("stream_out");

  Stream_setData (stream_transaction,
                  transaction_setup,
                  transaction_setup_len);

  conv (CONV_SETUP, stream_transaction, stream_out, &debug);

  if (stream_transaction.empty())
  {
    printf("Data consume [PASS]\n");
  }
}

static void execution (unsigned int * input_tensor,
                       unsigned int input_tensor_len,
                       unsigned int * output_tensor,
                       unsigned int output_tensor_len)
{
  int debug;
  int num_samples;
  int num_words_channel = (DMA_CHANNEL_WIDTH / 8) / sizeof(unsigned int);
  ap_uint<DMA_CHANNEL_WIDTH> data;
  Data expected;
  Data output;
  float threshold_error = 0.50;
  float mse_error = 0;
  float mean_error = 0;
  float error = 0;
  float max_error = 0;

  // Send setup transaction
  printf("_________ Execution _________\n");

  hls::stream<StreamChannel> stream_input_tensor ("input_tensor");
  hls::stream<StreamChannel> stream_output_tensor ("output_tensor");

  Stream_setData(stream_input_tensor, input_tensor, input_tensor_len);

  conv (CONV_EXECUTION, stream_input_tensor, stream_output_tensor, &debug);

  if (stream_input_tensor.empty())
  {
    printf("Input tensor consumption [PASS]\n");
  }

  num_samples = output_tensor_len;

  for (unsigned int i = 0; i < num_samples; i += num_words_channel)
  {
    data = stream_output_tensor.read ().data;
    for (int word = 0; word < num_words_channel; word++)
    {
      output.u32 = data >> (8 * sizeof(unsigned int) * word);
      expected.u32 = output_tensor[i + word];

      if (output.f32 < expected.f32)
        error = expected.f32 - output.f32;
      else
        error = output.f32 - expected.f32;

      if (max_error < error)
        max_error = error;

      mse_error += pow(error, 2);
      mean_error += error;

      if (threshold_error <= error)
        printf (
            "Output tensor data mismatch [FAIL]: Index %d; Expected %f, Output %f\n",
            i + word, expected.f32, output.f32);
    }
  }

  mse_error /= num_samples;
  mean_error /= num_samples;

  printf ("Max error: %f\n", max_error);
  printf ("Mean error: %f\n", mean_error);
  printf ("Mean squared error (MSE): %f\n", mse_error);

  if (stream_input_tensor.empty ())
  {
    printf ("Output tensor content [PASS]\n");
    printf ("Output tensor consumption [PASS]\n");
  }
}

void run (TestCase * test_array, size_t test_array_len)
{
  for (auto i = 0; i < test_array_len; i ++)
  {
    printf ("(%d) Test case: %s\n", i, test_array[i].name);

    conv_test::setup (test_array[i].transaction_setup,
                      test_array[i].transaction_setup_len);

    conv_test::execution (test_array[i].input_tensor,
                          test_array[i].input_tensor_len,
                          test_array[i].output_tensor,
                          test_array[i].output_tensor_len);

    printf ("_____________________________________________________________\n");
  }
}

}


int main (void)
{
  printf("Test Bench\n");

  conv_test::TestCase test_array[] =
  {
    {
      .name                   = "CONV_2D",
      .transaction_setup      = conv_test::conv_2d::transaction_setup,
      .transaction_setup_len  = conv_test::conv_2d::transaction_setup_len,
      .input_tensor           = conv_test::conv_2d::input_tensor,
      .input_tensor_len       = conv_test::conv_2d::input_tensor_len,
      .output_tensor          = conv_test::conv_2d::output_tensor,
      .output_tensor_len      = conv_test::conv_2d::output_tensor_len
    },
    {
      .name                   = "DEPTHWISE_CONV_2D",
      .transaction_setup      = conv_test::depthwise_conv_2d::transaction_setup,
      .transaction_setup_len  = conv_test::depthwise_conv_2d::transaction_setup_len,
      .input_tensor           = conv_test::depthwise_conv_2d::input_tensor,
      .input_tensor_len       = conv_test::depthwise_conv_2d::input_tensor_len,
      .output_tensor          = conv_test::depthwise_conv_2d::output_tensor,
      .output_tensor_len      = conv_test::depthwise_conv_2d::output_tensor_len
    }
  };

  conv_test::run (test_array, sizeof(test_array) / sizeof(conv_test::TestCase));

  printf("DONE!\n");

  return 0;
}

