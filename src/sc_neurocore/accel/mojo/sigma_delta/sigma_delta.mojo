# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Build: mojo build --emit shared-lib -o libsigma_delta.so sigma_delta.mojo

from std.math import exp
from std.memory import UnsafePointer
def _finite(v:Float64)->Bool:return v==v and v<=1.7976931348623157e308 and v>=-1.7976931348623157e308
@export
def sigma_delta_simulate_c(steps:Int,sigma_init:Float64,reconstruction_init:Float64,delta:Float64,tau:Float64,dt:Float64,currents_addr:Int,sigma_addr:Int,reconstruction_addr:Int,events_addr:Int,sigma_final_addr:Int,reconstruction_final_addr:Int)->Int:
    """Run the complete configured sampled APSDM batch."""
    if steps<0:return 1
    var currents=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=currents_addr);var sigmas=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=sigma_addr);var reconstructions=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=reconstruction_addr);var events=UnsafePointer[Int64,MutAnyOrigin](unsafe_from_address=events_addr);var sigma_final=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=sigma_final_addr);var reconstruction_final=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=reconstruction_final_addr)
    var sigma=sigma_init;var reconstruction=reconstruction_init
    for i in range(steps):
        var current=currents[i]
        if not (_finite(sigma) and abs(sigma)<=1.0e12 and _finite(reconstruction) and abs(reconstruction)<=1.0e12 and _finite(delta) and delta>0.0 and _finite(tau) and tau>0.0 and _finite(dt) and dt>0.0 and _finite(current)):return 2
        var ns=sigma+dt*current;var nr=reconstruction*exp(-dt/tau);var event=Int64(0)
        if ns-nr>=0.5*delta:nr+=delta;event=Int64(1)
        if not (_finite(ns) and abs(ns)<=1.0e12 and _finite(nr) and abs(nr)<=1.0e12):return 2
        sigma=ns;reconstruction=nr;sigmas[i]=sigma;reconstructions[i]=reconstruction;events[i]=event
    sigma_final[0]=sigma;reconstruction_final[0]=reconstruction;return 0
