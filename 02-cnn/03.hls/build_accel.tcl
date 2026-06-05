# build_accel.tcl 


open_project -reset myproject_accel_prj

set_top myproject_mnist_accel

add_files firmware/myproject.cpp               -cflags "-std=c++14"
add_files firmware/myproject_mnist_accel.cpp  -cflags "-std=c++14"

add_files firmware/myproject.h
add_files firmware/myproject_mnist_accel.h
add_files firmware/parameters.h
add_files firmware/defines.h
add_files firmware/nnet_utils
add_files firmware/weights

open_solution -reset "solution1"

set_part {xczu3eg-sfvc784-2-e}

create_clock -period 10 -name default
config_compile  -name_max_length 80
config_schedule -enable_dsp_full_reg=false

# HLS synthesis
csynth_design

# Export IP to ./ip/
config_export -format ip_catalog -rtl verilog -output ./ip
export_design  -format ip_catalog -rtl verilog -output ./ip

exit
