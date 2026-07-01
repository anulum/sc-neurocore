// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Izhikevich neuron batch step kernel.
//
// One thread per neuron; each thread loops the full n_steps internally with a
// constant per-neuron input current, writing its own [n_steps] spike and voltage
// row. Mirrors the floating-point CPU `neuron::Izhikevich::step` exactly — two
// half-steps of the Euler update (for stability on the 0.04·v² term) then the
// v >= 30 threshold with reset v←c, u←u+d:
//
//   v' = 0.04·v² + 5·v + 140 − u + I ,  u' = a·(b·v − u)
//
// WGSL has no f64, so the math is f32 and agreement with the f64 CPU model is
// tolerance-based (spike counts and the sub-threshold trace), not bit-exact —
// unlike the fixed-point LIF kernel.

struct IzhParams {
    n_neurons: u32,
    n_steps: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    dt: f32,
    v_peak: f32,
};

@group(0) @binding(0) var<storage, read> currents: array<f32>;
@group(0) @binding(1) var<storage, read_write> spikes_out: array<i32>;
@group(0) @binding(2) var<storage, read_write> voltages_out: array<f32>;
@group(0) @binding(3) var<uniform> params: IzhParams;

@compute @workgroup_size(64)
fn izhikevich_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron: u32 = gid.x;
    if (neuron >= params.n_neurons) {
        return;
    }

    let i_t: f32 = currents[neuron];
    let half: f32 = params.dt * 0.5;
    var v: f32 = params.c;
    var u: f32 = params.b * params.c;
    let base: u32 = neuron * params.n_steps;

    for (var step: u32 = 0u; step < params.n_steps; step = step + 1u) {
        // Two half-steps, matching the CPU model's stability split.
        for (var sub: u32 = 0u; sub < 2u; sub = sub + 1u) {
            let dv: f32 = (0.04 * v * v + 5.0 * v + 140.0 - u + i_t) * half;
            let du: f32 = (params.a * (params.b * v - u)) * half;
            v = v + dv;
            u = u + du;
        }

        var spike: i32 = 0;
        if (v >= params.v_peak) {
            v = params.c;
            u = u + params.d;
            spike = 1;
        }
        spikes_out[base + step] = spike;
        voltages_out[base + step] = v;
    }
}
