#include "myproject_mnist_accel.h"
#include "myproject.h"

void myproject_mnist_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
) {
    #pragma HLS INTERFACE axis      port=s_axis_in
    #pragma HLS INTERFACE axis      port=s_axis_out
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS DATAFLOW

    // internal streams
    hls::stream<input_t>  core_in("core_in");
    hls::stream<result_t> core_out("core_out");

    #pragma HLS STREAM variable=core_in  depth=32
    #pragma HLS STREAM variable=core_out depth=32

INPUT_LOOP:
    for (int i = 0; i < MNIST_NSAMPLES; i++) {
        #pragma HLS PIPELINE II=1

        axis_t a = s_axis_in.read();

        ap_fixed<16,12> fx;
        fx.range(15,0) = a.data.range(15,0);  

        input_t pix;
        pix[0] = fx;

        core_in.write(pix);
    }


    myproject(core_in, core_out);


    result_t out_vec = core_out.read();


OUTPUT_LOOP:
    for (int c = 0; c < MNIST_NCLASSES; c++) {
        #pragma HLS PIPELINE II=1

        axis_t ao;

        ao.data.range(15,0)  = out_vec[c].range(15,0);
        ao.data.range(31,16) = 0;

        ao.keep = -1;
        ao.strb = -1;
        ao.user = 0;
        ao.id   = 0;
        ao.dest = 0;
        ao.last = (c == MNIST_NCLASSES - 1);

        s_axis_out.write(ao);
    }
}

