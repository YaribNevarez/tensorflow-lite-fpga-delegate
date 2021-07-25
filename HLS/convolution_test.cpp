#include <iostream>
#include <cstdlib>

#include "convolution.h"

using namespace std;

int main (void)
{
  int debug;
  ConvExecutionMode mode = CONV_EXECUTION;

  hls::stream<StreamChannel> stream_in ("stream_in");
  hls::stream<StreamChannel> stream_out ("stream_out");

  conv (mode,stream_in, stream_out, &debug);

  return 0;
}

