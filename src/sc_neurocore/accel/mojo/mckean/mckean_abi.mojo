# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from std.memory import UnsafePointer
from mckean import mckean_next_v,mckean_next_w
@export
def mckean_simulate_c(steps:Int,v0:Float64,w0:Float64,a:Float64,l:Float64,mu:Float64,b:Float64,dt:Float64,currents_addr:Int,voltages_addr:Int,recovery_addr:Int,events_addr:Int,vfinal_addr:Int,wfinal_addr:Int)->Int:
    """Execute the complete right-continuous source RK4 batch."""
    if steps<0:return 1
    var currents=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=currents_addr);var voltages=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=voltages_addr);var recovery=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=recovery_addr);var events=UnsafePointer[Int64,MutAnyOrigin](unsafe_from_address=events_addr);var vfinal=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=vfinal_addr);var wfinal=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=wfinal_addr);var v=v0;var w=w0
    for i in range(steps):
        var nv=mckean_next_v(v,w,a,l,mu,b,dt,currents[i]);var nw=mckean_next_w(v,w,a,l,mu,b,dt,currents[i]);var event=1 if v<a and nv>=a else 0;v=nv;w=nw;voltages[i]=v;recovery[i]=w;events[i]=Int64(event)
    vfinal[0]=v;wfinal[0]=w;return 0
