#!/bin/bash

#
# Vivado(TM)
# runme.sh: a Vivado-generated Runs Script for UNIX
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#

if [ -z "$PATH" ]; then
  PATH=/tools/Xilinx/2025.2/Vitis/bin:/tools/Xilinx/2025.2/Vivado/ids_lite/ISE/bin/lin64:/tools/Xilinx/2025.2/Vivado/bin
else
  PATH=/tools/Xilinx/2025.2/Vitis/bin:/tools/Xilinx/2025.2/Vivado/ids_lite/ISE/bin/lin64:/tools/Xilinx/2025.2/Vivado/bin:$PATH
fi
export PATH

if [ -z "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=
else
  LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH

HD_PWD='/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE/sc_shd_pynq/sc_shd_pynq.runs/impl_1'
cd "$HD_PWD"

HD_LOG=runme.log
/bin/touch $HD_LOG

ISEStep="./ISEWrap.sh"
EAStep()
{
     $ISEStep $HD_LOG "$@" >> $HD_LOG 2>&1
     if [ $? -ne 0 ]
     then
         exit
     fi
}

# pre-commands:
/bin/touch .init_design.begin.rst
EAStep vivado -log system_wrapper.vdi -applog -m64 -product Vivado -messageDb vivado.pb -mode batch -source system_wrapper.tcl -notrace
