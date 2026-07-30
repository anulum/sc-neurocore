# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Build: mojo build --emit shared-lib -o libsc_sigma_delta_accumulator.so sc_sigma_delta_accumulator.mojo

from std.memory import UnsafePointer
def _finite(v:Float64)->Bool:return v==v and v<=1.7976931348623157e308 and v>=-1.7976931348623157e308
@export
def sc_sigma_delta_accumulator_simulate_c(steps:Int,sigma_init:Float64,threshold:Float64,currents_addr:Int,sigma_addr:Int,events_addr:Int,sigma_final_addr:Int)->Int:
    """Run the complete retained bipolar accumulator batch."""
    if steps<0:return 1
    var currents=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=currents_addr);var trace=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=sigma_addr);var events=UnsafePointer[Int64,MutAnyOrigin](unsafe_from_address=events_addr);var final=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=sigma_final_addr);var sigma=sigma_init
    for i in range(steps):
        var current=currents[i]
        if not (_finite(sigma) and _finite(threshold) and threshold>0.0 and _finite(current)):return 2
        var ns=sigma+current;var event=Int64(0)
        if ns>=threshold:ns-=threshold;event=Int64(1)
        elif ns<=-threshold:ns+=threshold;event=Int64(-1)
        if not _finite(ns):return 2
        sigma=ns;trace[i]=sigma;events[i]=event
    final[0]=sigma;return 0
