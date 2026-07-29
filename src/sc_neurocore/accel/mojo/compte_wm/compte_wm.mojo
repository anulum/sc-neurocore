# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for complete Compte batches

"""Source-bounded Compte pyramidal-cell batch kernel and stable C ABI."""

from std.math import exp, isfinite
from std.memory import UnsafePointer


@always_inline
def membrane_derivative(
    v: Float64, s_ampa: Float64, s_nmda: Float64, s_gaba: Float64,
    current: Float64, g_l: Float64, g_ampa: Float64, g_nmda: Float64,
    g_gaba: Float64, e_l: Float64, e_exc: Float64, e_inh: Float64,
    c_m: Float64, mg: Float64,
) -> Float64:
    """Evaluate the source pyramidal membrane derivative.

    Args:
        v: Membrane voltage in millivolts.
        s_ampa: External AMPA gate.
        s_nmda: Recurrent NMDA open fraction.
        s_gaba: Incoming GABAA gate.
        current: Direct somatic current in nanoamps.
        g_l: Leak conductance in microSiemens.
        g_ampa: External AMPA conductance in microSiemens.
        g_nmda: Recurrent NMDA conductance in microSiemens.
        g_gaba: Incoming GABAA conductance in microSiemens.
        e_l: Leak reversal in millivolts.
        e_exc: Excitatory reversal in millivolts.
        e_inh: Inhibitory reversal in millivolts.
        c_m: Membrane capacitance in nanofarads.
        mg: Extracellular magnesium concentration in millimolar.

    Returns:
        The membrane-voltage derivative in millivolts per millisecond.
    """
    var block = 1.0 / (1.0 + mg / 3.57 * exp(-0.062 * v))
    var i_l = g_l * (v - e_l)
    var i_ampa = g_ampa * s_ampa * (v - e_exc)
    var i_nmda = g_nmda * block * s_nmda * (v - e_exc)
    var i_gaba = g_gaba * s_gaba * (v - e_inh)
    return (-i_l - i_ampa - i_nmda - i_gaba + current) / c_m


@export
def compte_wm_simulate_c(
    steps: Int, v_init: Float64, s_ampa_init: Float64, s_nmda_init: Float64,
    x_nmda_init: Float64, s_gaba_init: Float64, ref_init: Float64,
    g_l: Float64, g_ampa: Float64, g_nmda: Float64, g_gaba: Float64,
    e_l: Float64, e_exc: Float64, e_inh: Float64, c_m: Float64, mg: Float64,
    tau_ampa: Float64, tau_nmda: Float64, tau_x: Float64, tau_gaba: Float64,
    alpha_nmda: Float64, v_threshold: Float64, v_reset: Float64,
    tau_ref: Float64, dt: Float64,
    currents_addr: Int, recurrent_addr: Int, external_addr: Int, inhibitory_addr: Int,
    voltages_addr: Int, s_ampa_out_addr: Int, s_nmda_out_addr: Int,
    x_nmda_out_addr: Int, s_gaba_out_addr: Int, refractory_addr: Int,
    events_addr: Int, v_final_addr: Int, s_ampa_final_addr: Int,
    s_nmda_final_addr: Int, x_nmda_final_addr: Int, s_gaba_final_addr: Int,
    ref_final_addr: Int,
) -> Int:
    """Run the complete source-level batch; nonzero status invalidates outputs.

    Args:
        steps: Number of physical timesteps.
        v_init: Initial membrane voltage.
        s_ampa_init: Initial external AMPA gate.
        s_nmda_init: Initial recurrent NMDA gate.
        x_nmda_init: Initial recurrent NMDA precursor.
        s_gaba_init: Initial incoming GABAA gate.
        ref_init: Initial refractory time remaining.
        g_l: Leak conductance.
        g_ampa: External AMPA conductance.
        g_nmda: Recurrent NMDA conductance.
        g_gaba: Incoming GABAA conductance.
        e_l: Leak reversal potential.
        e_exc: Excitatory reversal potential.
        e_inh: Inhibitory reversal potential.
        c_m: Membrane capacitance.
        mg: Extracellular magnesium concentration.
        tau_ampa: AMPA decay time.
        tau_nmda: NMDA open-fraction decay time.
        tau_x: NMDA precursor decay time.
        tau_gaba: GABAA decay time.
        alpha_nmda: NMDA saturation rate.
        v_threshold: Sampled spike threshold.
        v_reset: Reset voltage.
        tau_ref: Absolute refractory interval.
        dt: Physical timestep.
        currents_addr: Address of the binary64 current input array.
        recurrent_addr: Address of the recurrent-event Int64 array.
        external_addr: Address of the external-event Int64 array.
        inhibitory_addr: Address of the inhibitory-event Int64 array.
        voltages_addr: Address of the voltage output array.
        s_ampa_out_addr: Address of the AMPA output array.
        s_nmda_out_addr: Address of the NMDA output array.
        x_nmda_out_addr: Address of the NMDA-precursor output array.
        s_gaba_out_addr: Address of the GABAA output array.
        refractory_addr: Address of the refractory output array.
        events_addr: Address of the sampled output-event array.
        v_final_addr: Address of the final-voltage scalar output.
        s_ampa_final_addr: Address of the final-AMPA scalar output.
        s_nmda_final_addr: Address of the final-NMDA scalar output.
        x_nmda_final_addr: Address of the final-precursor scalar output.
        s_gaba_final_addr: Address of the final-GABAA scalar output.
        ref_final_addr: Address of the final-refractory scalar output.

    Returns:
        Zero on success; nonzero means validation failed before outputs became valid.
    """
    if steps < 0:
        return 1
    if not (
        isfinite(v_init) and v_init >= -200.0 and v_init <= 100.0
        and isfinite(s_ampa_init) and s_ampa_init >= 0.0 and s_ampa_init <= 1.0e6
        and isfinite(s_nmda_init) and s_nmda_init >= 0.0 and s_nmda_init <= 1.0
        and isfinite(x_nmda_init) and x_nmda_init >= 0.0 and x_nmda_init <= 1.0e6
        and isfinite(s_gaba_init) and s_gaba_init >= 0.0 and s_gaba_init <= 1.0e6
        and isfinite(ref_init) and ref_init >= 0.0
        and isfinite(g_l) and g_l >= 0.0 and isfinite(g_ampa) and g_ampa >= 0.0
        and isfinite(g_nmda) and g_nmda >= 0.0 and isfinite(g_gaba) and g_gaba >= 0.0
        and isfinite(e_l) and isfinite(e_exc) and isfinite(e_inh)
        and isfinite(c_m) and c_m > 0.0 and isfinite(mg) and mg >= 0.0
        and isfinite(tau_ampa) and tau_ampa > 0.0
        and isfinite(tau_nmda) and tau_nmda > 0.0
        and isfinite(tau_x) and tau_x > 0.0
        and isfinite(tau_gaba) and tau_gaba > 0.0
        and isfinite(alpha_nmda) and alpha_nmda >= 0.0
        and isfinite(v_threshold) and isfinite(v_reset)
        and v_reset >= -200.0 and v_reset <= 100.0
        and isfinite(tau_ref) and tau_ref > 0.0 and isfinite(dt) and dt > 0.0
    ):
        return 2
    var currents = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=currents_addr)
    var recurrent = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=recurrent_addr)
    var external = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=external_addr)
    var inhibitory = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=inhibitory_addr)
    for index in range(steps):
        if not isfinite(currents[index]):
            return 3
        if not (
            (recurrent[index] == 0 or recurrent[index] == 1)
            and (external[index] == 0 or external[index] == 1)
            and (inhibitory[index] == 0 or inhibitory[index] == 1)
        ):
            return 3
    var voltages = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=voltages_addr)
    var s_ampa_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=s_ampa_out_addr)
    var s_nmda_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=s_nmda_out_addr)
    var x_nmda_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_nmda_out_addr)
    var s_gaba_out = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=s_gaba_out_addr)
    var refractory = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=refractory_addr)
    var events = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=events_addr)
    var v_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=v_final_addr)
    var s_ampa_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=s_ampa_final_addr)
    var s_nmda_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=s_nmda_final_addr)
    var x_nmda_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=x_nmda_final_addr)
    var s_gaba_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=s_gaba_final_addr)
    var ref_final = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=ref_final_addr)
    var v = v_init
    var s_ampa = s_ampa_init
    var s_nmda = s_nmda_init
    var x_nmda = x_nmda_init
    var s_gaba = s_gaba_init
    var ref_remaining = ref_init
    for index in range(steps):
        if external[index] == 1:
            s_ampa += 1.0
        if recurrent[index] == 1:
            x_nmda += 1.0
        if inhibitory[index] == 1:
            s_gaba += 1.0
        if not (
            s_ampa >= 0.0 and s_ampa <= 1.0e6 and s_nmda >= 0.0 and s_nmda <= 1.0
            and x_nmda >= 0.0 and x_nmda <= 1.0e6 and s_gaba >= 0.0 and s_gaba <= 1.0e6
        ):
            return 4
        var active = ref_remaining <= 0.0
        var k1_v = 0.0
        if active:
            k1_v = membrane_derivative(
                v, s_ampa, s_nmda, s_gaba, currents[index], g_l, g_ampa,
                g_nmda, g_gaba, e_l, e_exc, e_inh, c_m, mg,
            )
        var k1_ampa = -s_ampa / tau_ampa
        var k1_nmda = -s_nmda / tau_nmda + alpha_nmda * x_nmda * (1.0 - s_nmda)
        var k1_x = -x_nmda / tau_x
        var k1_gaba = -s_gaba / tau_gaba
        var mid_v = v + 0.5 * dt * k1_v
        var mid_ampa = s_ampa + 0.5 * dt * k1_ampa
        var mid_nmda = s_nmda + 0.5 * dt * k1_nmda
        var mid_x = x_nmda + 0.5 * dt * k1_x
        var mid_gaba = s_gaba + 0.5 * dt * k1_gaba
        var k2_v = 0.0
        if active:
            k2_v = membrane_derivative(
                mid_v, mid_ampa, mid_nmda, mid_gaba, currents[index], g_l,
                g_ampa, g_nmda, g_gaba, e_l, e_exc, e_inh, c_m, mg,
            )
        var next_v = v + dt * k2_v
        var next_ampa = s_ampa + dt * (-mid_ampa / tau_ampa)
        var next_nmda = s_nmda + dt * (
            -mid_nmda / tau_nmda + alpha_nmda * mid_x * (1.0 - mid_nmda)
        )
        var next_x = x_nmda + dt * (-mid_x / tau_x)
        var next_gaba = s_gaba + dt * (-mid_gaba / tau_gaba)
        if not (
            isfinite(next_v) and next_v >= -200.0 and next_v <= 100.0
            and isfinite(next_ampa) and next_ampa >= 0.0 and next_ampa <= 1.0e6
            and isfinite(next_nmda) and next_nmda >= 0.0 and next_nmda <= 1.0
            and isfinite(next_x) and next_x >= 0.0 and next_x <= 1.0e6
            and isfinite(next_gaba) and next_gaba >= 0.0 and next_gaba <= 1.0e6
        ):
            return 4
        var event = Int64(0)
        ref_remaining = max(0.0, ref_remaining - dt)
        if not active:
            next_v = v_reset
        elif next_v >= v_threshold:
            next_v = v_reset
            ref_remaining = tau_ref
            event = 1
        v, s_ampa, s_nmda, x_nmda, s_gaba = (
            next_v, next_ampa, next_nmda, next_x, next_gaba
        )
        voltages[index] = v
        s_ampa_out[index] = s_ampa
        s_nmda_out[index] = s_nmda
        x_nmda_out[index] = x_nmda
        s_gaba_out[index] = s_gaba
        refractory[index] = ref_remaining
        events[index] = event
    v_final[0] = v
    s_ampa_final[0] = s_ampa
    s_nmda_final[0] = s_nmda
    x_nmda_final[0] = x_nmda
    s_gaba_final[0] = s_gaba
    ref_final[0] = ref_remaining
    return 0

# Build: mojo build --emit shared-lib -o libcompte_wm.so compte_wm.mojo
