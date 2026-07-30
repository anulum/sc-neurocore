# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Fardet-Levina eLIF RK4 kernel

def _energy_lif_finite(x:Float64)->Bool:return x==x and x<=1.7976931348623157e308 and x>=-1.7976931348623157e308
def _energy_lif_rhs_v(v:Float64,e:Float64,current:Float64,c:Float64,g:Float64,e0:Float64,eu:Float64,epsilon0:Float64)->Float64:
    var leak=e0+(eu-e0)*(1.0-e/epsilon0);return (g*(leak-v)+current)/c
def _energy_lif_rhs_e(v:Float64,e:Float64,alpha:Float64,epsilon0:Float64,ed:Float64,ef:Float64,tau_e:Float64)->Float64:
    var x=1.0-e/(alpha*epsilon0);return (x*x*x-(v-ef)/(ed-ef))/tau_e
def energy_lif_valid(v:Float64,e:Float64,c:Float64,g:Float64,e0:Float64,eu:Float64,ed:Float64,ef:Float64,vth:Float64,vreset:Float64,alpha:Float64,epsilon0:Float64,epsilon_c:Float64,delta:Float64,tau_e:Float64,dt:Float64)->Bool:
    return _energy_lif_finite(v) and v>=-200.0 and v<=100.0 and _energy_lif_finite(e) and e>=0.0 and e<=5.0 and _energy_lif_finite(c) and c>0.0 and _energy_lif_finite(g) and g>0.0 and _energy_lif_finite(e0) and _energy_lif_finite(eu) and _energy_lif_finite(ed) and _energy_lif_finite(ef) and ed!=ef and _energy_lif_finite(vth) and _energy_lif_finite(vreset) and vreset>=-200.0 and vreset<=100.0 and vth>vreset and _energy_lif_finite(alpha) and alpha>0.0 and _energy_lif_finite(epsilon0) and epsilon0>0.0 and _energy_lif_finite(epsilon_c) and epsilon_c>=0.0 and _energy_lif_finite(delta) and delta>=0.0 and _energy_lif_finite(tau_e) and tau_e>0.0 and _energy_lif_finite(dt) and dt>0.0 and dt<=1.0 and dt<=tau_e
def energy_lif_next_v(v:Float64,e:Float64,c:Float64,g:Float64,e0:Float64,eu:Float64,ed:Float64,ef:Float64,vth:Float64,vreset:Float64,alpha:Float64,epsilon0:Float64,epsilon_c:Float64,delta:Float64,tau_e:Float64,dt:Float64,current:Float64)->Float64:
    if not (energy_lif_valid(v,e,c,g,e0,eu,ed,ef,vth,vreset,alpha,epsilon0,epsilon_c,delta,tau_e,dt) and _energy_lif_finite(current)):return 0.0/0.0
    var k1v=_energy_lif_rhs_v(v,e,current,c,g,e0,eu,epsilon0);var k1e=_energy_lif_rhs_e(v,e,alpha,epsilon0,ed,ef,tau_e)
    var k2v=_energy_lif_rhs_v(v+dt*k1v/2.0,e+dt*k1e/2.0,current,c,g,e0,eu,epsilon0);var k2e=_energy_lif_rhs_e(v+dt*k1v/2.0,e+dt*k1e/2.0,alpha,epsilon0,ed,ef,tau_e)
    var k3v=_energy_lif_rhs_v(v+dt*k2v/2.0,e+dt*k2e/2.0,current,c,g,e0,eu,epsilon0);var k3e=_energy_lif_rhs_e(v+dt*k2v/2.0,e+dt*k2e/2.0,alpha,epsilon0,ed,ef,tau_e)
    var k4v=_energy_lif_rhs_v(v+dt*k3v,e+dt*k3e,current,c,g,e0,eu,epsilon0)
    return v+dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
def energy_lif_next_epsilon(v:Float64,e:Float64,c:Float64,g:Float64,e0:Float64,eu:Float64,ed:Float64,ef:Float64,vth:Float64,vreset:Float64,alpha:Float64,epsilon0:Float64,epsilon_c:Float64,delta:Float64,tau_e:Float64,dt:Float64,current:Float64)->Float64:
    if not (energy_lif_valid(v,e,c,g,e0,eu,ed,ef,vth,vreset,alpha,epsilon0,epsilon_c,delta,tau_e,dt) and _energy_lif_finite(current)):return 0.0/0.0
    var k1v=_energy_lif_rhs_v(v,e,current,c,g,e0,eu,epsilon0);var k1e=_energy_lif_rhs_e(v,e,alpha,epsilon0,ed,ef,tau_e)
    var k2v=_energy_lif_rhs_v(v+dt*k1v/2.0,e+dt*k1e/2.0,current,c,g,e0,eu,epsilon0);var k2e=_energy_lif_rhs_e(v+dt*k1v/2.0,e+dt*k1e/2.0,alpha,epsilon0,ed,ef,tau_e)
    var k3v=_energy_lif_rhs_v(v+dt*k2v/2.0,e+dt*k2e/2.0,current,c,g,e0,eu,epsilon0);var k3e=_energy_lif_rhs_e(v+dt*k2v/2.0,e+dt*k2e/2.0,alpha,epsilon0,ed,ef,tau_e)
    var k4e=_energy_lif_rhs_e(v+dt*k3v,e+dt*k3e,alpha,epsilon0,ed,ef,tau_e)
    return e+dt*(k1e+2.0*k2e+2.0*k3e+k4e)/6.0
def energy_lif_step_spike(v:Float64,e:Float64,c:Float64,g:Float64,e0:Float64,eu:Float64,ed:Float64,ef:Float64,vth:Float64,vreset:Float64,alpha:Float64,epsilon0:Float64,epsilon_c:Float64,delta:Float64,tau_e:Float64,dt:Float64,current:Float64)->Int:
    var nv=energy_lif_next_v(v,e,c,g,e0,eu,ed,ef,vth,vreset,alpha,epsilon0,epsilon_c,delta,tau_e,dt,current);var ne=energy_lif_next_epsilon(v,e,c,g,e0,eu,ed,ef,vth,vreset,alpha,epsilon0,epsilon_c,delta,tau_e,dt,current)
    if not (_energy_lif_finite(nv) and nv>=-200.0 and nv<=100.0 and _energy_lif_finite(ne) and ne>=0.0 and ne<=5.0):return -1
    if nv>vth and ne>epsilon_c:
        if ne-delta<0.0:return -1
        return 1
    return 0
