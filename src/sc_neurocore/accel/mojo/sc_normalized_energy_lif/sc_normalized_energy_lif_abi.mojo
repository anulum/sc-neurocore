# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Build: mojo build -I ../kernels --emit shared-lib -o libsc_normalized_energy_lif.so sc_normalized_energy_lif_abi.mojo
from std.memory import UnsafePointer
from sc_normalized_energy_lif import sc_normalized_energy_lif_next_epsilon,sc_normalized_energy_lif_next_v,sc_normalized_energy_lif_step_spike
@export
def sc_normalized_energy_lif_simulate_c(steps:Int,v0:Float64,e0s:Float64,vrest:Float64,vreset:Float64,vth:Float64,tau_m:Float64,tau_e:Float64,alpha:Float64,epsilon0:Float64,resistance:Float64,dt:Float64,currents_addr:Int,voltages_addr:Int,energies_addr:Int,events_addr:Int,vfinal_addr:Int,efinal_addr:Int)->Int:
    """Execute the complete retained SC batch through the C ABI."""
    if steps<0:return 1
    var currents=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=currents_addr);var voltages=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=voltages_addr);var energies=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=energies_addr);var events=UnsafePointer[Int64,MutAnyOrigin](unsafe_from_address=events_addr);var vfinal=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=vfinal_addr);var efinal=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=efinal_addr);var v=v0;var e=e0s
    for i in range(steps):
        var current=currents[i];var event=sc_normalized_energy_lif_step_spike(v,e,vrest,vreset,vth,tau_m,tau_e,alpha,epsilon0,resistance,dt,current);var nv=sc_normalized_energy_lif_next_v(v,e,vrest,tau_m,tau_e,epsilon0,resistance,dt,current);var ne=sc_normalized_energy_lif_next_epsilon(e,epsilon0,tau_e,dt)
        if event<0:return 2
        if event==1:v=vreset;e=max(0.0,ne-alpha)
        else:v=nv;e=ne
        voltages[i]=v;energies[i]=e;events[i]=Int64(event)
    vfinal[0]=v;efinal[0]=e;return 0
