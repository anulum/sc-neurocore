// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Fixed-point LIF batch step kernel.
//
// One thread per neuron; each thread loops the full n_steps internally with a
// constant per-neuron input current, writing its own [n_steps] spike and voltage
// row. This mirrors the CPU `batch_lif_run_multi` contract (constant current,
// zero noise) and is a bit-exact integer port of `FixedPointLif::step` — WGSL
// signed `>>` is arithmetic, so the leak/gain shifts and the `mask` helper match
// the Rust two's-complement arithmetic exactly.

struct LifParams {
    n_neurons: u32,
    n_steps: u32,
    data_width: u32,
    fraction: u32,
    leak_k: i32,
    gain_k: i32,
    v_rest: i32,
    v_reset: i32,
    v_threshold: i32,
    refractory_period: i32,
    noise: i32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> currents: array<i32>;
@group(0) @binding(1) var<storage, read_write> spikes_out: array<i32>;
@group(0) @binding(2) var<storage, read_write> voltages_out: array<i32>;
@group(0) @binding(3) var<uniform> params: LifParams;

// Port of `neuron::mask(value, width) as i16`, returned widened to i32.
// Masks the low `width` bits with sign-extension, then truncates to a signed
// 16-bit value (the Rust return type), which the caller uses as i32.
fn mask(value: i32, width: u32) -> i32 {
    var x: i32;
    if (width >= 32u) {
        x = value;
    } else {
        let m: u32 = (1u << width) - 1u;
        let masked: u32 = bitcast<u32>(value) & m;
        let sh: u32 = 32u - width;
        x = (bitcast<i32>(masked) << sh) >> sh;
    }
    // Emulate `as i16`: keep low 16 bits, sign-extend bit 15.
    let lo: u32 = bitcast<u32>(x) & 0xFFFFu;
    return (bitcast<i32>(lo) << 16u) >> 16u;
}

@compute @workgroup_size(64)
fn lif_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron: u32 = gid.x;
    if (neuron >= params.n_neurons) {
        return;
    }

    let w: u32 = params.data_width;
    let frac: u32 = params.fraction;
    let i_t: i32 = currents[neuron];
    let noise: i32 = params.noise;

    var v: i32 = params.v_rest;
    var refractory_counter: i32 = 0;
    let base: u32 = neuron * params.n_steps;

    for (var step: u32 = 0u; step < params.n_steps; step = step + 1u) {
        var spike: i32 = 0;
        var volt: i32 = 0;
        if (refractory_counter > 0) {
            refractory_counter = refractory_counter - 1;
            v = params.v_rest;
            spike = 0;
            volt = mask(params.v_rest, w);
        } else {
            let diff: i32 = mask(params.v_rest - v, 2u * w);
            let dv_leak: i32 = mask((diff * params.leak_k) >> frac, w);
            let dv_in: i32 = mask((i_t * params.gain_k) >> frac, w);
            let v_next: i32 = mask(v + dv_leak + dv_in + noise, w);
            if (v_next >= params.v_threshold) {
                v = params.v_reset;
                refractory_counter = params.refractory_period;
                spike = 1;
                volt = mask(params.v_reset, w);
            } else {
                v = v_next;
                spike = 0;
                volt = mask(v_next, w);
            }
        }
        spikes_out[base + step] = spike;
        voltages_out[base + step] = volt;
    }
}
