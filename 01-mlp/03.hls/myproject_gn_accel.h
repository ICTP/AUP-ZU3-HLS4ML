
#pragma once

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "myproject.h"

typedef ap_axiu<32, 1, 1, 1> axis_t;

void myproject_gn_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
);
