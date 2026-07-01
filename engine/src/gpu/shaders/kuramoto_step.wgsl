// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Kuramoto oscillator Euler step kernel.
//
// One thread per oscillator; each thread reads the whole phase vector and sums
// its coupling row K_nm * sin(theta_m - theta_n) over all m, so the O(N^2)
// all-to-all coupling is fully parallel across oscillators (this is where the GPU
// wins over the rayon CPU path). One dispatch advances one Euler step; the host
// ping-pongs phase_in/phase_out between steps. This mirrors the baseline update
// of `scpn::kuramoto::KuramotoSolver::step` with zero noise:
//
//   dtheta_n = omega_n + (sum_m K_nm * sin(theta_m - theta_n)) / N
//   theta_n  = rem_euclid(theta_n + dtheta_n * dt, 2*pi)
//
// WGSL has no f64, so the arithmetic is f32 and agreement with the f64 CPU
// oracle is tolerance-based (the coupling sum order and libm sin also differ),
// not bit-exact — unlike the fixed-point LIF kernel.

const TWO_PI: f32 = 6.2831853071795864769;

struct KuramotoParams {
    n_osc: u32,
    _pad: u32,
    dt: f32,
    n_inv: f32,
};

@group(0) @binding(0) var<storage, read> omega: array<f32>;
@group(0) @binding(1) var<storage, read> coupling: array<f32>;
@group(0) @binding(2) var<storage, read> phase_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> phase_out: array<f32>;
@group(0) @binding(4) var<uniform> params: KuramotoParams;

@compute @workgroup_size(64)
fn kuramoto_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n: u32 = gid.x;
    if (n >= params.n_osc) {
        return;
    }

    let theta_n: f32 = phase_in[n];
    let row: u32 = n * params.n_osc;
    var coupling_sum: f32 = 0.0;
    for (var m: u32 = 0u; m < params.n_osc; m = m + 1u) {
        coupling_sum = coupling_sum + coupling[row + m] * sin(phase_in[m] - theta_n);
    }

    let dtheta: f32 = omega[n] + coupling_sum * params.n_inv;
    var next: f32 = theta_n + dtheta * params.dt;
    // rem_euclid(next, TWO_PI): floor-based wrap gives a result in [0, 2*pi).
    next = next - floor(next / TWO_PI) * TWO_PI;
    phase_out[n] = next;
}
