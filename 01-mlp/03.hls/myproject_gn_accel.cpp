
#include "myproject_gn_accel.h"

static const int GN_NSAMPLES = 161;
static const int GN_NCLASSES = 2;

void myproject_gn_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
) {
#pragma HLS INTERFACE axis      port=s_axis_in
#pragma HLS INTERFACE axis      port=s_axis_out
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS DATAFLOW

    hls::stream<input_t>  core_in("core_in");
    hls::stream<result_t> core_out("core_out");
#pragma HLS STREAM variable=core_in  depth=2
#pragma HLS STREAM variable=core_out depth=2

    input_t in_vec;
GN_INPUT_LOOP:
    for (int i = 0; i < GN_NSAMPLES; i++) {
    #pragma HLS PIPELINE II=1
        axis_t a = s_axis_in.read();
        in_vec[i] = a.data;
    }

    core_in.write(in_vec);
    myproject(core_in, core_out);

    result_t out_vec = core_out.read();

GN_OUTPUT_LOOP:
    for (int c = 0; c < GN_NCLASSES; c++) {
    #pragma HLS PIPELINE II=1
        axis_t a_out;
        a_out.data = out_vec[c];
        a_out.keep = -1;
        a_out.strb = -1;
        a_out.user = 0;
        a_out.id   = 0;
        a_out.dest = 0;
        a_out.last = (c == GN_NCLASSES - 1);
        s_axis_out.write(a_out);
    }
}
