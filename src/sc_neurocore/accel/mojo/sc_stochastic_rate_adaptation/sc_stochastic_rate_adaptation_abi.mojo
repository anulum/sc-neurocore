# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from std.memory import UnsafePointer
from std.math import exp
fn rate(a:Float64,current:Float64,fmax:Float64,beta:Float64,ihalf:Float64)->Float64:
    var z=beta*(current-a-ihalf)
    if z>=0.0:return fmax/(1.0+exp(-z))
    var ez=exp(z);return fmax*ez/(1.0+ez)
fn rhs(a:Float64,current:Float64,fmax:Float64,beta:Float64,ihalf:Float64,tau:Float64,delta:Float64)->Float64:
    return -a/tau+delta*rate(a,current,fmax,beta,ihalf)
@export
def sc_sra_simulate_c(steps:Int,a0:Float64,fmax:Float64,beta:Float64,ihalf:Float64,tau:Float64,delta:Float64,dt:Float64,currents_addr:Int,uniforms_addr:Int,adaptation_addr:Int,events_addr:Int,afinal_addr:Int)->Int:
    var currents=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=currents_addr);var uniforms=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=uniforms_addr);var adaptation=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=adaptation_addr);var events=UnsafePointer[Int64,MutAnyOrigin](unsafe_from_address=events_addr);var afinal=UnsafePointer[Float64,MutAnyOrigin](unsafe_from_address=afinal_addr);var a=a0
    for i in range(steps):
        var k1=rhs(a,currents[i],fmax,beta,ihalf,tau,delta);var r1=rate(a,currents[i],fmax,beta,ihalf);var a2=a+0.5*dt*k1;var k2=rhs(a2,currents[i],fmax,beta,ihalf,tau,delta);var r2=rate(a2,currents[i],fmax,beta,ihalf);var a3=a+0.5*dt*k2;var k3=rhs(a3,currents[i],fmax,beta,ihalf,tau,delta);var r3=rate(a3,currents[i],fmax,beta,ihalf);var a4=a+dt*k3;var k4=rhs(a4,currents[i],fmax,beta,ihalf,tau,delta);var r4=rate(a4,currents[i],fmax,beta,ihalf);a=a+dt/6.0*(k1+2.0*k2+2.0*k3+k4);var p=1.0-exp(-(r1+2.0*r2+2.0*r3+r4)/6.0*dt/1000.0);adaptation[i]=a;events[i]=Int64(1 if uniforms[i]<p else 0)
    afinal[0]=a;return 0
