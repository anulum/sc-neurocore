# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header
# Build: mojo build --emit shared-lib -o libbenda_herz.so benda_herz_abi.mojo

from std.memory import UnsafePointer
from std.math import sqrt

def rate(a:Float64,current:Float64,gain:Float64,rheo:Float64)->Float64:
    return gain*sqrt(max(current-a-rheo,0.0))

def rhs_a(a:Float64,current:Float64,gain:Float64,rheo:Float64,slope:Float64,tau:Float64)->Float64:
    return (slope*rate(a,current,gain,rheo)-a)/tau

def rhs_p(a:Float64,current:Float64,gain:Float64,rheo:Float64)->Float64:
    return rate(a,current,gain,rheo)/1000.0

@export
def benda_herz_simulate_c(steps:Int,a0:Float64,p0:Float64,gain:Float64,rheo:Float64,slope:Float64,tau:Float64,dt:Float64,currents_addr:Int,adaptation_addr:Int,phases_addr:Int,events_addr:Int,afinal_addr:Int,pfinal_addr:Int)->Int:
    if steps<0:return 1
    var currents=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=currents_addr);var adaptation=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=adaptation_addr);var phases=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=phases_addr);var events=UnsafePointer[Int64,MutAnyOrigin](unsafe_from_address=events_addr);var afinal=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=afinal_addr);var pfinal=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=pfinal_addr);var a=a0;var phase=p0
    for i in range(steps):
        var k1a=rhs_a(a,currents[i],gain,rheo,slope,tau);var k1p=rhs_p(a,currents[i],gain,rheo);var k2a=rhs_a(a+0.5*dt*k1a,currents[i],gain,rheo,slope,tau);var k2p=rhs_p(a+0.5*dt*k1a,currents[i],gain,rheo);var k3a=rhs_a(a+0.5*dt*k2a,currents[i],gain,rheo,slope,tau);var k3p=rhs_p(a+0.5*dt*k2a,currents[i],gain,rheo);var k4a=rhs_a(a+dt*k3a,currents[i],gain,rheo,slope,tau);var k4p=rhs_p(a+dt*k3a,currents[i],gain,rheo);var scale=dt/6.0;a=a+scale*(k1a+2.0*k2a+2.0*k3a+k4a);phase=phase+scale*(k1p+2.0*k2p+2.0*k3p+k4p);var event:Int64=0
        if phase>=1.0:phase=0.0;event=1
        adaptation[i]=a;phases[i]=phase;events[i]=event
    afinal[0]=a;pfinal[0]=phase;return 0
