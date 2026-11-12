# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for Brunel-Wang midpoint-RK2 batches

from std.math import exp, isfinite
from std.memory import Pointer


@always_inline
def derivative(
    voltage: Float64, ext: Float64, ampa: Float64, nmda: Float64, gaba: Float64,
    v_rest: Float64, tau_m: Float64, g_ampa_ext: Float64, g_ampa_rec: Float64,
    g_nmda: Float64, g_gaba: Float64, v_ampa: Float64, v_nmda: Float64,
    v_gaba: Float64, c_m: Float64, mg_conc: Float64,
) -> Float64:
    """Evaluate the paper membrane derivative for fixed aggregate gates."""
    var block = 1.0 / (1.0 + mg_conc / 3.57 * exp(-0.062 * voltage))
    var i_ampa = -g_ampa_ext * (voltage - v_ampa) * ext - g_ampa_rec * (voltage - v_ampa) * ampa
    var i_nmda = -g_nmda * block * (voltage - v_nmda) * nmda
    var i_gaba = -g_gaba * (voltage - v_gaba) * gaba
    return -(voltage - v_rest) / tau_m + (i_ampa + i_nmda + i_gaba) / c_m


# The caller owns stable Float64 aggregate gates[steps] (external AMPA, recurrent
# AMPA, NMDA, GABA), disjoint Float64 voltage and refractory traces[steps], an
# Int64 event trace[steps], and writable Float64 final voltage/refractory scalars.
# Addresses, extents and non-overlap are caller preconditions; the scalar outputs
# must also be writable for zero steps. No pointer ownership survives the call.
# Status 0 completes the batch; 1 rejects negative steps, 2 configuration,
# 3 nonfinite/negative gates, 4 nonfinite dynamics. A dynamics refusal can leave
# partial traces: callers must discard every output when status is nonzero.
@export
def brunel_wang_simulate_c(
    steps: Int, v_init: Float64, ref_init: Float64, v_rest: Float64,
    v_reset: Float64, v_threshold: Float64, tau_m: Float64, tau_ref: Float64,
    g_ampa_ext: Float64, g_ampa_rec: Float64, g_nmda: Float64, g_gaba: Float64,
    v_ampa: Float64, v_nmda: Float64, v_gaba: Float64, c_m: Float64,
    mg_conc: Float64, dt: Float64, ext_addr: Int, ampa_addr: Int, nmda_addr: Int,
    gaba_addr: Int, voltages_addr: Int, refractory_addr: Int, events_addr: Int,
    v_final_addr: Int, ref_final_addr: Int,
) abi("C") -> Int:
    """Run one complete configured batch; nonzero status invalidates outputs."""
    if steps < 0:
        return 1
    if not (
        isfinite(v_init) and isfinite(ref_init) and ref_init >= 0.0
        and isfinite(v_rest) and isfinite(v_reset) and isfinite(v_threshold)
        and isfinite(tau_m) and tau_m > 0.0 and isfinite(tau_ref) and tau_ref > 0.0
        and isfinite(g_ampa_ext) and g_ampa_ext >= 0.0
        and isfinite(g_ampa_rec) and g_ampa_rec >= 0.0
        and isfinite(g_nmda) and g_nmda >= 0.0 and isfinite(g_gaba) and g_gaba >= 0.0
        and isfinite(v_ampa) and isfinite(v_nmda) and isfinite(v_gaba)
        and isfinite(c_m) and c_m > 0.0 and isfinite(mg_conc) and mg_conc >= 0.0
        and isfinite(dt) and dt > 0.0
    ):
        return 2
    var ext = Pointer[Float64, MutAnyOrigin](unsafe_from_address=ext_addr)
    var ampa = Pointer[Float64, MutAnyOrigin](unsafe_from_address=ampa_addr)
    var nmda = Pointer[Float64, MutAnyOrigin](unsafe_from_address=nmda_addr)
    var gaba = Pointer[Float64, MutAnyOrigin](unsafe_from_address=gaba_addr)
    var voltages = Pointer[Float64, MutAnyOrigin](unsafe_from_address=voltages_addr)
    var refractory = Pointer[Float64, MutAnyOrigin](unsafe_from_address=refractory_addr)
    var events = Pointer[Int64, MutAnyOrigin](unsafe_from_address=events_addr)
    var v_final = Pointer[Float64, MutAnyOrigin](unsafe_from_address=v_final_addr)
    var ref_final = Pointer[Float64, MutAnyOrigin](unsafe_from_address=ref_final_addr)
    for index in range(steps):
        if not (isfinite(ext[unsafe_offset=index]) and ext[unsafe_offset=index] >= 0.0 and isfinite(ampa[unsafe_offset=index]) and ampa[unsafe_offset=index] >= 0.0 and isfinite(nmda[unsafe_offset=index]) and nmda[unsafe_offset=index] >= 0.0 and isfinite(gaba[unsafe_offset=index]) and gaba[unsafe_offset=index] >= 0.0):
            return 3
    var v = v_init
    var ref_remaining = ref_init
    for index in range(steps):
        var event = Int64(0)
        if ref_remaining > 0.0:
            v = v_reset
            ref_remaining = max(0.0, ref_remaining - dt)
        else:
            var k1 = derivative(v, ext[unsafe_offset=index], ampa[unsafe_offset=index], nmda[unsafe_offset=index], gaba[unsafe_offset=index], v_rest, tau_m, g_ampa_ext, g_ampa_rec, g_nmda, g_gaba, v_ampa, v_nmda, v_gaba, c_m, mg_conc)
            var midpoint = v + 0.5 * dt * k1
            var k2 = derivative(midpoint, ext[unsafe_offset=index], ampa[unsafe_offset=index], nmda[unsafe_offset=index], gaba[unsafe_offset=index], v_rest, tau_m, g_ampa_ext, g_ampa_rec, g_nmda, g_gaba, v_ampa, v_nmda, v_gaba, c_m, mg_conc)
            var candidate = v + dt * k2
            if not (isfinite(k1) and isfinite(midpoint) and isfinite(k2) and isfinite(candidate)):
                return 4
            v = candidate
            if candidate >= v_threshold:
                v = v_reset
                ref_remaining = tau_ref
                event = 1
        voltages[unsafe_offset=index] = v
        refractory[unsafe_offset=index] = ref_remaining
        events[unsafe_offset=index] = event
    v_final[unsafe_offset=0] = v
    ref_final[unsafe_offset=0] = ref_remaining
    return 0

# Build: mojo build --Werror --target-cpu x86-64-v3 --emit shared-lib -o libbrunel_wang.so brunel_wang.mojo
