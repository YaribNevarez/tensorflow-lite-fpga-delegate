#include <iostream>
#include <cstdlib>

#include "convolution.h"
#include "data_test.h"
#include "cmath"

using namespace std;



void Stream_setData(hls::stream<StreamChannel> & stream, unsigned int * data, unsigned int len)
{
  StreamChannel channel;
  for (unsigned int i = 0; i < len; i += 2)
  {
    channel.data = (0xFFFFFFFF00000000 & (((ap_uint<DMA_CHANNEL_WIDTH> ) data[i + 1]) << 32))
              | (0x00000000FFFFFFFF & data[i]);
    stream.write (channel);
  }
}

namespace conv_hw
{
  namespace test
  {
    void setup (void)
    {
      int debug;
      // Send setup transaction
      printf("1.- CONV_LOAD_PROFILE_PACKAGE\n");

      hls::stream<StreamChannel> stream_transaction ("transaction");
      hls::stream<StreamChannel> stream_out ("stream_out");

      Stream_setData(stream_transaction, transaction_setup, transaction_setup_len);

      conv (CONV_LOAD_PROFILE_PACKAGE, stream_transaction, stream_out, &debug);

      if (stream_transaction.empty())
      {
        printf("Data consume PASS\n");
      }
    }

    void execution (void)
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
      printf("2.- CONV_EXECUTION\n");

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
  }
}

int main (void)
{
  printf("CONV test bench\n");

  conv_hw::test::setup();

  conv_hw::test::execution();

  printf("DONE!\n");

  return 0;
}

