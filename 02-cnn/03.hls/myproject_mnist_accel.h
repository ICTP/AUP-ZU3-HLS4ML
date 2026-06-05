#ifndef MYPROJECT_MNIST_ACCEL_H_
#define MYPROJECT_MNIST_ACCEL_H_

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "myproject.h"

#define MNIST_DATA_W   32
#define MNIST_NSAMPLES 784
#define MNIST_NCLASSES 10

typedef ap_axiu<MNIST_DATA_W,1,1,1> axis_t;

void myproject_mnist_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
);

#endif

