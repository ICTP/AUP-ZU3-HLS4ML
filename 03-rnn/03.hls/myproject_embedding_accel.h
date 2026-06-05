
#ifndef MYPROJECT_EMBEDDING_ACCEL_H_
#define MYPROJECT_EMBEDDING_ACCEL_H_

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "myproject.h"


#define EMB_DATA_W     32
#define EMB_NSAMPLES   30     
#define EMB_NCLASSES   100    

typedef ap_axiu<EMB_DATA_W,1,1,1> axis_t;

void myproject_embedding_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
);

#endif

