
#include "myproject_embedding_accel.h"
#include "myproject.h"

void myproject_embedding_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
) {
#pragma HLS INTERFACE axis      port=s_axis_in
#pragma HLS INTERFACE axis      port=s_axis_out
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS DATAFLOW

    hls::stream<integer_input_t> core_in("core_in");
    hls::stream<result_t>        core_out("core_out");

#pragma HLS STREAM variable=core_in  depth=4
#pragma HLS STREAM variable=core_out depth=4


    integer_input_t token;   

READ_INPUT:
    for (int i = 0; i < EMB_NSAMPLES; i++) {
#pragma HLS PIPELINE II=1
        axis_t a = s_axis_in.read();

        ap_uint<16> id = a.data.range(15,0);

        token[i] = (ap_fixed<16,10>) id;
    }

    // push FULL vector into HLS core
    core_in.write(token);

    // run model
    myproject(core_in, core_out);

    // read output
    result_t out_vec = core_out.read();

WRITE_OUTPUT:
    for (int c = 0; c < EMB_NCLASSES; c++) {
#pragma HLS PIPELINE II=1
        axis_t ao;

        ao.data.range(15,0)  = out_vec[c].range(15,0);
        ao.data.range(31,16) = 0;

        ao.keep = -1;
        ao.strb = -1;
        ao.user = 0;
        ao.id   = 0;
        ao.dest = 0;
        ao.last = (c == EMB_NCLASSES - 1);

        s_axis_out.write(ao);
    }
}

